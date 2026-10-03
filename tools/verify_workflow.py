"""Execute real isolated ComfyUI workflows, optionally through cloud bridges."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
TYPES = {
    "1": "UNIMATE_ASSET",
    "2": "UNIMATE_RIG",
    "3": "UNIMATE_MODEL",
    "4": "UNIMATE_MOTION",
}


def validate_output(filename, data):
    if not data:
        raise ValueError("Empty exported artifact")
    if filename.endswith(".glb"):
        from unimate_pack.assets import validate_glb
        validate_glb(data)
    elif filename.endswith('.fbx'):
        if not data.startswith(b'Kaydara FBX Binary  \x00\x1a\x00'):
            raise ValueError('Invalid binary FBX artifact header')
    elif filename.endswith(".npz"):
        from unimate_pack.motion_io import load_motion
        load_motion(data)
    elif filename.endswith('.png'):
        import io
        from PIL import Image
        try:
            with Image.open(io.BytesIO(data)) as image:
                if image.format != 'PNG' or not (0 < image.width <= 1024 and 0 < image.height <= 1024):
                    raise ValueError('Unexpected skeleton image dimensions or format')
                image.load()
        except OSError as error:
            raise ValueError('Invalid preview PNG') from error
    else:
        if json.loads(data).get("schema") != "unimate.export.v1":
            raise ValueError("Unexpected export metadata schema")


def validate_batch_cases(cases, prompts, repetitions):
    expected = [(i, prompt, 60) for i, prompt in enumerate(
        prompt for prompt in prompts for _ in range(repetitions))]
    actual = sorted((case["seed"], case["prompt"], case["frames"]) for case in cases)
    if actual != expected:
        raise ValueError("Batch export provenance does not match the requested cases")


def validate_multi_rig_batch_cases(cases, prompts, repetitions, sources):
    if not sources or len(set(sources)) != len(sources):
        raise ValueError('Multi-rig fixture requires distinct source assets')
    expected = [(i, prompt, 60, source) for i, (source, prompt) in enumerate(
        (source, prompt) for source in sources for prompt in prompts for _ in range(repetitions))]
    actual = sorted((case['seed'], case['prompt'], case['frames'], case['source_sha256']) for case in cases)
    if actual != expected:
        raise ValueError('Multi-rig batch provenance has missing or misassigned cases')
    identities = [{case['rig_id'] for case in cases if case['source_sha256'] == source} for source in sources]
    if any(len(group) != 1 for group in identities) or len(set.union(*identities)) != len(sources):
        raise ValueError('Multi-rig batch does not retain distinct, stable rig identities')


def bundle_inventory(path):
    """Compare restored artifacts without loading multi-GB model bytes again."""
    with zipfile.ZipFile(path) as archive:
        result = {}
        for name in archive.namelist():
            digest = hashlib.sha256()
            with archive.open(name) as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            result[name] = digest.hexdigest()
        return result


def request(base, route, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        base + route, data=data, headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        raise RuntimeError(error.read().decode(errors="replace")) from error


def submit(base, graph, timeout=7200):
    queued = json.loads(request(base, "/prompt", {"prompt": graph}))
    prompt_id = queued["prompt_id"]
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        history = json.loads(request(base, "/history/" + prompt_id))
        if prompt_id in history:
            entry = history[prompt_id]
            if entry["status"]["status_str"] != "success":
                errors = [
                    {
                        key: value
                        for key, value in message[1].items()
                        if key
                        in (
                            "node_id",
                            "node_type",
                            "exception_message",
                            "exception_type",
                            "traceback",
                        )
                    }
                    for message in entry["status"].get("messages", [])
                    if message[0] == "execution_error"
                ]
                raise RuntimeError(json.dumps(errors, indent=2))
            return entry
        time.sleep(0.5)
    raise TimeoutError(f"ComfyUI prompt {prompt_id} exceeded {timeout}s")


def install_link(source, target):
    if os.name == "nt":
        # A directory junction avoids requiring Windows symlink privileges.
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(target), str(source)], capture_output=True
        )
        if result.returncode:
            raise RuntimeError(
                result.stdout.decode(errors="replace")
                + result.stderr.decode(errors="replace")
            )
    else:
        target.symlink_to(source, target_is_directory=True)


def verify(args):
    if getattr(args, 'batch_worker', False) and not (
            getattr(args, 'multi_rig', False) and args.cloud_root and getattr(args, 'cloud_client_root', None)):
        raise ValueError('Batch worker verification requires multi-rig mode and cloud/client checkouts')
    if getattr(args, 'multi_rig', False) and not (getattr(args, 'batch', False) and args.branching):
        raise ValueError('Multi-rig verification requires batch mode and a branching first rig')
    if getattr(args, 'worker_lists', False) and not getattr(args, 'worker', False):
        raise ValueError('Execution-list verification requires worker mode')
    if getattr(args, 'worker', False) and not (
            args.cloud_root and getattr(args, 'cloud_client_root', None)
            and getattr(args, 'skeleton_reference', None)):
        raise ValueError('Worker verification requires cloud/client checkouts and a skeleton reference archive')
    workspace = args.workdir.resolve()
    if workspace.exists() and any(workspace.iterdir()):
        raise ValueError("Workflow verification requires a new empty directory")
    workspace.mkdir(parents=True, exist_ok=True)
    for name in (
        "input",
        "output",
        "temp",
        "user",
        "models/unimate",
        "custom_nodes",
        "partition",
    ):
        (workspace / name).mkdir(parents=True, exist_ok=True)
    install_link(ROOT, workspace / "custom_nodes" / "comfy-unimate")
    if args.cloud_root:
        bridge = (
            args.cloud_root
            / "deploy/runtime-profiles/comfyui/ComfyUI-Cloud-Offload-Runtime"
        )
        install_link(bridge.resolve(), workspace / "custom_nodes" / "cloud-runtime")
    bundle = workspace / "models/unimate" / args.bundle.name
    # A hard link stages the exact installed file without another multi-GB copy.
    try:
        os.link(args.bundle.resolve(), bundle)
    except OSError:
        shutil.copyfile(args.bundle, bundle)
    fixture = ROOT / "tests/fixtures/rig_generator.py"
    spec = importlib.util.spec_from_file_location("workflow_fixture", fixture)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    (workspace / "input" / "rig with spaces.glb").write_bytes(
        module.synthetic_glb(args.branching)
    )
    graph = json.loads((ROOT / "examples/unimate_api.json").read_text())
    graph["1"]["inputs"]["asset"] = "rig with spaces.glb"
    graph["3"]["inputs"]["bundle"] = bundle.name
    graph["5"]["inputs"]["filename_prefix"] = "verified/local"
    if getattr(args, 'skeleton_reference', None):
        if (args.cloud_root and not getattr(args, 'worker', False)) or getattr(args, 'extended', False) or getattr(args, 'batch', False):
            raise ValueError('Run archive skeleton verification separately from inference or bridges')
        shutil.copyfile(args.skeleton_reference, workspace / 'input/reference.npz')
        graph['4'] = {'class_type': 'UniMateLoadMotion', 'inputs': {'archive': 'reference.npz'}}
        graph.pop('3')
        for method in ('fk', 'ric'):
            graph[f'recover_{method}'] = {'class_type': 'UniMateRecoverSkeleton', 'inputs': {
                'rig': ['2', 0], 'motion': ['4', 0], 'method': method}}
            graph[f'preview_{method}'] = {'class_type': 'UniMatePreviewSkeleton', 'inputs': {
                'skeleton': [f'recover_{method}', 0], 'projection': 'front', 'resolution': 128}}
            graph[f'save_{method}'] = {'class_type': 'SaveImage', 'inputs': {
                'images': [f'preview_{method}', 0], 'filename_prefix': f'verified/{method}'}}
    if getattr(args, "batch", False):
        if (args.cloud_root and not getattr(args, 'batch_worker', False)) or getattr(args, "extended", False):
            raise ValueError("Run batch verification separately from bridge or expanded verification")
        graph["4"] = {"class_type": "UniMateGenerateBatch", "inputs": {
            "model": ["3", 0], "rig": ["2", 0], "seed": 0, "guidance": 3.0,
            "normalization": "objaverse", "repetitions": 2,
            "prompts": json.dumps(["A character stands still.", "A character walks forward."])}}
        if getattr(args, 'multi_rig', False):
            (workspace / 'input/chain rig.glb').write_bytes(module.synthetic_glb(False))
            graph['6'] = {'class_type': 'UniMateLoadRig', 'inputs': {'asset': 'chain rig.glb'}}
            graph['7'] = {'class_type': 'UniMatePrepareRig', 'inputs': {'asset': ['6', 0], 'facing': '+Z',
                'left_joint': '', 'right_joint': ''}}
            graph['8'] = {'class_type': 'UniMateCombineRigs', 'inputs': {'rig_a': ['2', 0], 'rig_b': ['7', 0]}}
            graph['4']['inputs']['rig'] = ['8', 0]
            graph['5']['inputs']['rig'] = ['4', 1]
    if getattr(args, "extended", False):
        for mode, selection in (("inbetween", "0,-1"), ("edit", "Joint_1")):
            graph[mode] = {
                "class_type": "UniMateInbetweenMotion" if mode == "inbetween" else "UniMateEditMotion",
                "inputs": {"model": ["3", 0], "rig": ["2", 0], "reference": ["4", 0],
                    "prompt": "A character walks forward.", "seed": 1, "guidance": 3.0,
                    "selection": selection, "normalization": "objaverse"},
            }
        graph["expand"] = {"class_type": "UniMateExpandMotion", "inputs": {
            "model": ["3", 0], "rig": ["2", 0], "seed": 2, "guidance": 3.0,
            "normalization": "objaverse", "overlap": 10,
            "prompts": json.dumps(["A character stands still.", "A character walks forward."])}}
        for mode in ("inbetween", "edit", "expand"):
            graph[mode + "_export"] = {"class_type": "UniMateExportGLB", "inputs": {
                "rig": ["2", 0], "motion": [mode, 0], "filename_prefix": "verified/" + mode}}
            graph[mode + "_save"] = {"class_type": "UniMateSaveMotion", "inputs": {
                "motion": [mode, 0], "filename_prefix": "verified/" + mode}}
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    base = f"http://127.0.0.1:{port}"
    env = os.environ.copy()
    env.update(
        UNIMATE_BLENDER=str(args.blender.resolve()),
        COMFY_PARTITION_ROOT=str(workspace / "partition"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONUNBUFFERED="1",
    )
    if args.cloud_root:
        env["PYTHONPATH"] = (
            str(args.cloud_root.resolve()) + os.pathsep + env.get("PYTHONPATH", "")
        )
    command = [
        # Resolving a Unix venv symlink selects the system interpreter instead.
        str(args.python.absolute()),
        str(args.comfy_root.resolve() / "main.py"),
        "--listen",
        "127.0.0.1",
        "--port",
        str(port),
        "--base-directory",
        str(workspace),
        "--user-directory",
        str(workspace / "user"),
        "--database-url",
        "sqlite:///:memory:",
        "--disable-api-nodes",
        "--disable-auto-launch",
        "--disable-all-custom-nodes",
        "--whitelist-custom-nodes",
        "comfy-unimate",
        "cloud-runtime",
    ]
    if getattr(args, "cpu", False):
        command.append("--cpu")
    revision = subprocess.run(
        ["git", "-C", str(args.comfy_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    report = {
        "comfy_root": str(args.comfy_root),
        "comfy_revision": revision,
        "workspace": str(workspace),
        "python_executable": str(args.python.absolute()),
        "cpu_requested": bool(getattr(args, "cpu", False)),
        "graphs": [],
    }
    log = (workspace / "server.log").open("wb")
    process = subprocess.Popen(
        command,
        cwd=args.comfy_root,
        env=env,
        stdout=log,
        stderr=subprocess.STDOUT,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    (workspace / "server.log").read_text(errors="replace")[-10000:]
                )
            try:
                info = json.loads(request(base, "/object_info"))
                break
            except (OSError, RuntimeError):
                time.sleep(0.5)
        else:
            raise TimeoutError("Isolated ComfyUI did not start")
        assert all(graph[n]["class_type"] in info for n in graph), (
            "UniMate nodes failed registration"
        )
        report["system_stats"] = json.loads(request(base, "/system_stats"))
        if getattr(args, 'batch_worker', False):
            sys.path.insert(0, str(args.comfy_root.resolve()))
            from tools.cloud_batch_workflow import run_batch_worker_workflows
            entries, evidence = run_batch_worker_workflows(base, graph, workspace, args.cloud_root,
                                                          args.cloud_client_root, bundle)
            report['worker_execution'] = evidence
            report['graphs'].extend({'kind': 'real-batch-worker-partition', 'graph': job['partition']['workflow'],
                'status': job['status']} for job in evidence['jobs'])
        elif getattr(args, 'worker', False):
            sys.path.insert(0, str(args.comfy_root.resolve()))
            from tools.cloud_workflow import run_worker_workflows
            entries, evidence = run_worker_workflows(base, graph, workspace, args.cloud_root,
                                                     args.cloud_client_root,
                                                     getattr(args, 'worker_lists', False))
            report['worker_execution'] = evidence
            report['graphs'].extend({'kind': 'real-worker-partition', 'graph': job['partition']['workflow'],
                'status': job['status']} for job in evidence['jobs'])
        elif args.cloud_root:
            # Capture actual node values through ComfyUI execution, never direct codec calls.
            capture = json.loads(json.dumps(graph))
            for node, type_name in TYPES.items():
                capture["out_" + node] = {
                    "class_type": "CloudPartitionOutput",
                    "inputs": {
                        "value": [node, 0],
                        "boundary_key": node,
                        "output_path": str(workspace / "partition" / (node + ".zip")),
                        "type_name": type_name,
                    },
                }
            entry = submit(base, capture)
            report["graphs"].append(
                {
                    "kind": "local-and-boundary-capture",
                    "graph": capture,
                    "status": entry["status"],
                }
            )
            # Restore every type independently, then run preparation, generation and export.
            boundary = {}
            for node, type_name in TYPES.items():
                boundary["in_" + node] = {
                    "class_type": "CloudPartitionInput",
                    "inputs": {
                        "boundary_key": node,
                        "artifact_path": str(workspace / "partition" / (node + ".zip")),
                        "type_name": type_name,
                    },
                }
                boundary["round_" + node] = {
                    "class_type": "CloudPartitionOutput",
                    "inputs": {
                        "value": ["in_" + node, 0],
                        "boundary_key": node,
                        "output_path": str(
                            workspace / "partition" / (node + "-round.zip")
                        ),
                        "type_name": type_name,
                    },
                }
            boundary["prepare"] = {
                "class_type": "UniMatePrepareRig",
                "inputs": {"asset": ["in_1", 0], "facing": "+Z"},
            }
            boundary["prepared"] = {
                "class_type": "CloudPartitionOutput",
                "inputs": {
                    "value": ["prepare", 0],
                    "boundary_key": "prepared",
                    "output_path": str(workspace / "partition/prepared.zip"),
                    "type_name": "UNIMATE_RIG",
                },
            }
            boundary["generate"] = {
                "class_type": "UniMateGenerateMotion",
                "inputs": {
                    **graph["4"]["inputs"],
                    "rig": ["in_2", 0],
                    "model": ["in_3", 0],
                },
            }
            boundary["export"] = {
                "class_type": "UniMateExportGLB",
                "inputs": {
                    "rig": ["in_2", 0],
                    "motion": ["in_4", 0],
                    "filename_prefix": "verified/cloud-restored",
                },
            }
            boundary["generated_export"] = {
                "class_type": "UniMateExportGLB",
                "inputs": {
                    "rig": ["in_2", 0],
                    "motion": ["generate", 0],
                    "filename_prefix": "verified/cloud-generated",
                },
            }
            entry2 = submit(base, boundary)
            for node, type_name in TYPES.items():
                assert bundle_inventory(
                    workspace / "partition" / (node + ".zip")
                ) == bundle_inventory(
                    workspace / "partition" / (node + "-round.zip")
                ), type_name + " changed across actual bridge execution"
            report["boundary_types_verified"] = list(TYPES.values())
            report["graphs"].append(
                {
                    "kind": "restored-types-consumed-by-real-nodes",
                    "graph": boundary,
                    "status": entry2["status"],
                }
            )
            entries = [entry, entry2]
        else:
            entry = submit(base, graph)
            report["graphs"].append(
                {"kind": ('archive-skeleton' if getattr(args, 'skeleton_reference', None)
                          else 'local-batch' if getattr(args, 'batch', False)
                          else 'local-expanded' if getattr(args, 'extended', False)
                          else 'local-five-nodes'), "graph": graph, "status": entry["status"]}
            )
            entries = [entry]
        exports = []
        batch_cases = []
        for entry in entries:
            for output in entry["outputs"].values():
                for key in ("3d", "files", "images"):
                    for file in output.get(key, []):
                        data = request(base, "/view?" + urllib.parse.urlencode(file))
                        validate_output(file["filename"], data)
                        if getattr(args, "batch", False) and file["filename"].endswith(".json"):
                            manifest = json.loads(data)
                            batch_cases.append({**manifest['generation'], 'rig_id': manifest['rig_id'],
                                                'source_sha256': manifest['source_sha256']})
                        exports.append({**file, "bytes": len(data)})
        assert len(exports) >= 2
        if getattr(args, 'skeleton_reference', None):
            from unimate_pack.motion_io import load_motion
            from unimate_pack.contracts import decode_arrays
            frames = len(decode_arrays(load_motion(args.skeleton_reference.read_bytes())['features'])['features'])
            assert sum(f['filename'].endswith('.png') for f in exports) == frames * (
                10 if getattr(args, 'worker_lists', False) else 6 if getattr(args, 'worker', False) else 2)
            report['skeleton_frames_per_mode'] = frames
        if getattr(args, "batch", False):
            multi_rig = getattr(args, 'multi_rig', False)
            jobs = 2 if getattr(args, 'batch_worker', False) else 1
            assert sum(f["filename"].endswith(".glb") for f in exports) == (8 if multi_rig else 4) * jobs
            assert len(exports) == (16 if multi_rig else 8) * jobs
            prompts = json.loads(graph['4']['inputs']['prompts'])
            if multi_rig:
                sources = [hashlib.sha256((workspace / 'input' / name).read_bytes()).hexdigest()
                           for name in ('rig with spaces.glb', 'chain rig.glb')]
                for offset in range(0, len(batch_cases), 8):
                    validate_multi_rig_batch_cases(batch_cases[offset:offset + 8], prompts, 2, sources)
                report['batch_source_sha256'] = sources
            else:
                validate_batch_cases(batch_cases, prompts, 2)
            report["batch_cases_verified"] = batch_cases
        report["retrieved_outputs"] = exports
        report["status"] = "passed"
        (workspace / "report.json").write_text(
            json.dumps(report, indent=2), encoding="utf8"
        )
        return report
    finally:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comfy-root", required=True, type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--blender", required=True, type=Path)
    parser.add_argument(
        "--workdir", required=True, type=Path, help="New isolated disposable directory"
    )
    parser.add_argument(
        "--cloud-root",
        type=Path,
        help="Cloud Offload source checkout with runner bridge nodes",
    )
    parser.add_argument("--branching", action="store_true")
    parser.add_argument("--extended", action="store_true", help="Exercise constrained modes, expansion and numeric saving")
    parser.add_argument("--batch", action="store_true", help="Exercise typed motion-list export for four cases")
    parser.add_argument('--multi-rig', action='store_true', help='Generate eight paired cases on two topologies; requires --batch --branching')
    parser.add_argument('--batch-worker', action='store_true', help='Stage models/assets and run paired batch worker jobs; requires multi-rig/cloud/client options')
    parser.add_argument('--skeleton-reference', type=Path, help='Verify archive loading, both recovery modes and every preview frame')
    parser.add_argument('--worker', action='store_true', help='Execute preprocessing/recovery through the real partition worker')
    parser.add_argument('--worker-lists', action='store_true', help='Also verify two distinct mapped motion cases; requires --worker')
    parser.add_argument('--cloud-client-root', type=Path, help='Cloud Offload node checkout for client artifact restoration')
    parser.add_argument("--cpu", action="store_true", help="Run ComfyUI on CPU")
    args = parser.parse_args()
    print(json.dumps(verify(args), indent=2))


if __name__ == "__main__":
    main()
