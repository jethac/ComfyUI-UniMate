"""ComfyUI schemas, managed files, and delegation to UniMate subsystems."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import shutil
import tempfile
import uuid

import folder_paths
from comfy_api.latest import io


Asset = io.Custom("UNIMATE_ASSET")
Rig = io.Custom("UNIMATE_RIG")
Model = io.Custom("UNIMATE_MODEL")
Motion = io.Custom("UNIMATE_MOTION")
CATEGORY = "3D/UniMate"
FACING = ["+Z", "-Z", "+X", "-X", "joint_pair"]
NORMALIZATION = ["objaverse", "mixamo", "truebones"]
MAX_ASSET_BYTES = 256 * 1024 * 1024

# Register only this bundle format; respect paths already configured by ComfyUI.
folder_paths.add_model_folder_path(
    "unimate", os.path.join(folder_paths.models_dir, "unimate")
)
model_paths, model_extensions = folder_paths.folder_names_and_paths["unimate"]
folder_paths.folder_names_and_paths["unimate"] = (
    model_paths,
    set(model_extensions) | {".unimate"},
)


def _relative_name(value: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError("UniMate requires a nonempty relative filename.")
    if (
        PurePosixPath(value).is_absolute()
        or PureWindowsPath(value).drive
        or PureWindowsPath(value).is_absolute()
    ):
        raise ValueError("UniMate filenames must be relative to a managed directory.")
    parts = value.replace("\\", "/").split("/")
    if any(
        part in ("", ".", "..")
        or part.endswith((".", " "))
        or any(ord(char) < 32 or char in '<>:"|?*' for char in part)
        for part in parts
    ):
        raise ValueError("UniMate filename contains an unsafe path component.")
    return "/".join(parts)


def _contained(path: Path, root: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("UniMate file resolves outside its permitted directory.")
    return resolved


def _input_path(asset: str) -> Path:
    if not isinstance(asset, str):
        raise ValueError("UniMate asset must be a selected GLB filename.")
    name, annotated_root = folder_paths.annotated_filepath(asset)
    if (
        annotated_root is not None
        and Path(annotated_root).resolve()
        != Path(folder_paths.get_input_directory()).resolve()
    ):
        raise ValueError("UniMate rig assets must be in ComfyUI input.")
    name = _relative_name(name)
    if Path(name).suffix.lower() != ".glb":
        raise ValueError("UniMate supports self-contained rigged .glb files only.")
    root = Path(folder_paths.get_input_directory())
    path = _contained(Path(folder_paths.get_annotated_filepath(name, str(root))), root)
    if not path.is_file():
        raise FileNotFoundError(f"UniMate input GLB is missing: {name}")
    if not 0 < path.stat().st_size <= MAX_ASSET_BYTES:
        raise ValueError("UniMate input GLB is empty or exceeds the 256 MiB limit.")
    return path


def _model_path(bundle: str) -> Path:
    name = _relative_name(bundle)
    if Path(name).suffix.lower() != ".unimate":
        raise ValueError("Select an installed .unimate model bundle.")
    path = folder_paths.get_full_path_or_raise("unimate", name)
    resolved = Path(path).resolve()
    if not any(
        resolved.is_relative_to(Path(root).resolve())
        for root in folder_paths.get_folder_paths("unimate")
    ):
        raise ValueError(
            "UniMate bundle resolves outside its registered model directories."
        )
    return resolved


def _fingerprint(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_choices() -> list[str]:
    root = Path(folder_paths.get_input_directory())
    result = []
    if root.is_dir():
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() == ".glb":
                name = path.relative_to(root).as_posix()
                try:
                    _input_path(name)
                except (ValueError, OSError):
                    continue
                result.append(name)
    return sorted(result)


class UniMateLoadRig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UniMateLoadRig",
            display_name="Load Rigged GLB",
            category=CATEGORY,
            inputs=[io.Combo.Input("asset", options=_input_choices())],
            outputs=[Asset.Output()],
        )

    @classmethod
    def execute(cls, asset: str) -> io.NodeOutput:
        path = _input_path(asset)
        from .unimate_pack.assets import validate_glb
        from .unimate_pack.contracts import make_asset

        with path.open("rb") as stream:
            glb = stream.read(MAX_ASSET_BYTES + 1)
        if len(glb) > MAX_ASSET_BYTES:
            raise ValueError("UniMate input GLB exceeds the 256 MiB limit.")
        validate_glb(glb)
        return io.NodeOutput(make_asset(glb, path.name))

    @classmethod
    def validate_inputs(cls, asset):
        try:
            _input_path(asset)
        except (ValueError, OSError) as error:
            return str(error)
        return True

    @classmethod
    def fingerprint_inputs(cls, asset):
        return _fingerprint(_input_path(asset))

    @classmethod
    def cloud_offload_assets(cls, inputs):
        selected = inputs.get("asset")
        if not isinstance(selected, str):
            raise ValueError("UniMate Cloud Offload requires a selected input GLB.")
        _input_path(selected)
        name, _ = folder_paths.annotated_filepath(selected)
        return [{"category": "__input__", "filename": _relative_name(name)}]


class UniMatePrepareRig(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UniMatePrepareRig",
            display_name="Prepare UniMate Rig",
            category=CATEGORY,
            inputs=[
                Asset.Input("asset"),
                io.Combo.Input("facing", options=FACING, default="+Z"),
                io.String.Input("left_joint", default="", optional=True),
                io.String.Input("right_joint", default="", optional=True),
            ],
            outputs=[Rig.Output()],
        )

    @classmethod
    def execute(cls, asset, facing, left_joint="", right_joint="") -> io.NodeOutput:
        from .unimate_pack.blender import prepare_rig
        from .unimate_pack.contracts import validate_asset, validate_rig

        validate_asset(asset)
        if facing not in FACING:
            raise ValueError("Select a supported UniMate facing direction.")
        if facing == "joint_pair" and (
            not left_joint or not right_joint or left_joint == right_joint
        ):
            raise ValueError(
                "Joint-pair facing requires distinct left_joint and right_joint names."
            )
        rig = prepare_rig(asset, facing, left_joint, right_joint)
        validate_rig(rig)
        return io.NodeOutput(rig)


class UniMateModelLoader(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        bundles = [
            name
            for name in folder_paths.get_filename_list("unimate")
            if Path(name).suffix.lower() == ".unimate"
        ]
        return io.Schema(
            node_id="UniMateModelLoader",
            display_name="Load UniMate Model",
            category=CATEGORY,
            inputs=[io.Combo.Input("bundle", options=bundles)],
            outputs=[Model.Output()],
        )

    @classmethod
    def execute(cls, bundle) -> io.NodeOutput:
        path = _model_path(bundle)
        from .unimate_pack.inference import load_model_bundle
        from .unimate_pack.contracts import validate_model

        model = load_model_bundle(path)
        validate_model(model)
        return io.NodeOutput(model)

    @classmethod
    def validate_inputs(cls, bundle):
        try:
            _model_path(bundle)
        except (ValueError, OSError) as error:
            return str(error)
        return True

    @classmethod
    def fingerprint_inputs(cls, bundle):
        return _fingerprint(_model_path(bundle))

    @classmethod
    def cloud_offload_assets(cls, inputs):
        selected = inputs.get("bundle")
        _model_path(selected)
        return [{"category": "unimate", "filename": _relative_name(selected)}]


class UniMateGenerateMotion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UniMateGenerateMotion",
            display_name="Generate UniMate Motion",
            category=CATEGORY,
            inputs=[
                Model.Input("model"),
                Rig.Input("rig"),
                io.String.Input("prompt", multiline=True),
                io.Int.Input(
                    "seed", default=0, min=0, max=2**64 - 1, control_after_generate=True
                ),
                io.Float.Input("guidance", default=3.0, min=1.0, max=10.0, step=0.1),
                io.Combo.Input(
                    "normalization", options=NORMALIZATION, default="objaverse"
                ),
            ],
            outputs=[Motion.Output()],
        )

    @classmethod
    def execute(
        cls, model, rig, prompt, seed, guidance, normalization="objaverse"
    ) -> io.NodeOutput:
        from .unimate_pack.inference import generate_motion
        from .unimate_pack.contracts import (
            validate_model,
            validate_rig,
            validate_motion,
        )

        validate_model(model)
        validate_rig(rig)
        if not isinstance(prompt, str):
            raise ValueError("UniMate prompt must be text.")
        if type(seed) is not int or not 0 <= seed <= 2**64 - 1:
            raise ValueError("UniMate seed must be an unsigned 64-bit integer.")
        if (
            not isinstance(guidance, (int, float))
            or not math.isfinite(guidance)
            or not 1.0 <= guidance <= 10.0
        ):
            raise ValueError("UniMate guidance must be finite and between 1 and 10.")
        if normalization not in NORMALIZATION:
            raise ValueError("Select objaverse, mixamo, or truebones normalization.")
        motion = generate_motion(
            model, rig, prompt, seed, guidance, normalization=normalization
        )
        validate_motion(motion, rig_id=rig["rig_id"])
        return io.NodeOutput(motion)


def _motion_input_path(archive):
    name = _relative_name(archive)
    if Path(name).suffix.lower() != ".npz":
        raise ValueError("Select a UniMate .npz motion archive")
    root = Path(folder_paths.get_input_directory()).resolve()
    path = _contained(root / name, root)
    if not path.is_file():
        raise FileNotFoundError("Motion archive is missing")
    if not 0 < path.stat().st_size <= MAX_ASSET_BYTES:
        raise ValueError("Motion archive is empty or exceeds 256 MiB")
    return path


class UniMateLoadMotion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        root = Path(folder_paths.get_input_directory())
        names = []
        for path in root.rglob("*.npz"):
            name = path.relative_to(root).as_posix()
            try:
                _motion_input_path(name)
            except (ValueError, OSError):
                continue
            names.append(name)
        return io.Schema(node_id=cls.__name__, display_name="Load UniMate Motion", category=CATEGORY,
                         inputs=[io.Combo.Input("archive", options=sorted(names))], outputs=[Motion.Output()])

    @classmethod
    def execute(cls, archive):
        from .unimate_pack.motion_io import load_motion
        with _motion_input_path(archive).open("rb") as stream:
            payload = stream.read(MAX_ASSET_BYTES + 1)
        if len(payload) > MAX_ASSET_BYTES:
            raise ValueError("Motion archive exceeds 256 MiB")
        return io.NodeOutput(load_motion(payload))

    @classmethod
    def fingerprint_inputs(cls, archive):
        return _fingerprint(_motion_input_path(archive))

    @classmethod
    def cloud_offload_assets(cls, inputs):
        archive = inputs.get("archive")
        _motion_input_path(archive)
        return [{"category": "__input__", "filename": _relative_name(archive)}]


class UniMateSaveMotion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Save UniMate Motion", category=CATEGORY,
                         inputs=[Motion.Input("motion"), io.String.Input("filename_prefix", default="unimate/motion")],
                         outputs=[Motion.Output()], is_output_node=True)

    @classmethod
    def execute(cls, motion, filename_prefix="unimate/motion"):
        from .unimate_pack.motion_io import dump_motion
        from comfy import model_management
        payload = dump_motion(motion)
        root = Path(folder_paths.get_output_directory()).resolve()
        prefix = _relative_name(filename_prefix)
        _contained(root / prefix, root)
        directory, filename, counter, _, _ = folder_paths.get_save_image_path(prefix, str(root))
        parent = _contained(Path(directory), root)
        final = _contained(parent / f"{filename}_{counter:05}_{uuid.uuid4().hex}.npz", root)
        descriptor, stage_name = tempfile.mkstemp(prefix=".unimate-motion-", dir=parent)
        stage = Path(stage_name)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            model_management.throw_exception_if_processing_interrupted()
            os.replace(stage, final)
        finally:
            stage.unlink(missing_ok=True)
        return io.NodeOutput(motion, ui={"files": [{"filename": final.name,
            "subfolder": parent.relative_to(root).as_posix(), "type": "output"}]})


class UniMateInbetweenMotion(io.ComfyNode):
    MODE = "inbetween"

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=cls.__name__,
            display_name="In-between UniMate Motion" if cls.MODE == "inbetween" else "Edit UniMate Motion",
            category=CATEGORY,
            inputs=[
                Model.Input("model"), Rig.Input("rig"), Motion.Input("reference"),
                io.String.Input("prompt", multiline=True),
                io.Int.Input("seed", default=0, min=0, max=2**64-1),
                io.Float.Input("guidance", default=3.0, min=1.1, max=10.0),
                io.String.Input("selection", default="0,-1" if cls.MODE == "inbetween" else "Hips",
                    tooltip="Comma-separated frame indices (negative from end) or original/clean joint names."),
                io.Combo.Input("normalization", options=NORMALIZATION, default="objaverse"),
            ], outputs=[Motion.Output()],
        )

    @classmethod
    def execute(cls, model, rig, reference, prompt, seed, guidance, selection, normalization="objaverse"):
        from .unimate_pack.inference import generate_motion
        return io.NodeOutput(generate_motion(
            model, rig, prompt, seed, guidance, normalization,
            reference=reference, constraint_mode=cls.MODE, selection=selection,
        ))


class UniMateEditMotion(UniMateInbetweenMotion):
    MODE = "edit"


class UniMateExpandMotion(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id=cls.__name__, display_name="Expand UniMate Motion", category=CATEGORY,
            inputs=[Model.Input("model"), Rig.Input("rig"),
                io.String.Input("prompts", default='["A character stands.", "A character walks forward."]', multiline=True,
                    tooltip="JSON array of prompts in segment order."),
                io.Int.Input("seed", default=0, min=0, max=2**64-1),
                io.Float.Input("guidance", default=3.0, min=1.1, max=10.0),
                io.Int.Input("overlap", default=10, min=1, max=59),
                io.Combo.Input("normalization", options=NORMALIZATION, default="objaverse")],
            outputs=[Motion.Output()])

    @classmethod
    def execute(cls, model, rig, prompts, seed, guidance, overlap=10, normalization="objaverse"):
        from .unimate_pack.expansion import expand_motion
        if not isinstance(prompts, str) or len(prompts) > 512 * 1024:
            raise ValueError("Prompt sequence must be a bounded JSON array")
        try:
            sequence = json.loads(prompts)
        except json.JSONDecodeError as error:
            raise ValueError("Prompt sequence must be a JSON array") from error
        return io.NodeOutput(expand_motion(model, rig, sequence, seed, guidance, normalization, overlap))


class UniMateExportGLB(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="UniMateExportGLB",
            display_name="Export UniMate GLB",
            category=CATEGORY,
            inputs=[
                Rig.Input("rig"),
                Motion.Input("motion"),
                io.String.Input("filename_prefix", default="unimate/animation"),
            ],
            outputs=[],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, rig, motion, filename_prefix="unimate/animation") -> io.NodeOutput:
        from .unimate_pack.contracts import validate_rig, validate_motion

        validate_rig(rig)
        validate_motion(motion, rig_id=rig["rig_id"])
        if motion.get("rig_id") != rig["rig_id"]:
            raise ValueError(
                "UniMate motion belongs to a different rig; regenerate it for this rig."
            )
        prefix = _relative_name(filename_prefix)
        output_root = Path(folder_paths.get_output_directory()).resolve()
        _contained(output_root / prefix, output_root)
        from .unimate_pack.blender import export_glb

        glb = export_glb(rig, motion)
        if not isinstance(glb, bytes) or not glb:
            raise ValueError("UniMate export did not return animated GLB bytes.")
        provenance = {
            "schema": "unimate.export.v1",
            "rig_id": rig["rig_id"],
            "source_sha256": rig["asset"]["sha256"],
            "features_sha256": hashlib.sha256(motion["features"]).hexdigest(),
            "glb_sha256": hashlib.sha256(glb).hexdigest(),
            "fps": motion["fps"],
            "generation": motion["metadata"],
        }
        # Serialize before staging so malformed provenance cannot publish an output.
        manifest = (
            json.dumps(provenance, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
        ).encode("utf-8")
        directory, filename, counter, _, _ = folder_paths.get_save_image_path(
            prefix, str(output_root)
        )
        parent = _contained(Path(directory), output_root)
        run_name = f"{filename}_{counter:05}_{uuid.uuid4().hex}"
        final = _contained(parent / run_name, output_root)
        stage = Path(tempfile.mkdtemp(prefix=".unimate-", dir=parent))
        try:
            for name, payload in (
                (filename + ".glb", glb),
                (filename + ".json", manifest),
            ):
                with (stage / name).open("xb") as stream:
                    stream.write(payload)
                    stream.flush()
                    os.fsync(stream.fileno())
            from comfy import model_management

            model_management.throw_exception_if_processing_interrupted()
            # A directory rename publishes the GLB/provenance pair together.
            os.replace(stage, final)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
        subfolder = final.relative_to(output_root).as_posix()
        return io.NodeOutput(
            ui={
                "3d": [
                    {
                        "filename": filename + ".glb",
                        "subfolder": subfolder,
                        "type": "output",
                    }
                ],
                "files": [
                    {
                        "filename": filename + ".json",
                        "subfolder": subfolder,
                        "type": "output",
                    }
                ],
            }
        )
