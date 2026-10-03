"""Node plumbing checks using ComfyUI's installed latest extension API."""

import asyncio
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = Path(os.environ.get("COMFYUI_ROOT", ROOT.parent / "ComfyUI"))
if not (COMFY_ROOT / "comfy_api" / "latest").is_dir():
    raise unittest.SkipTest("Set COMFYUI_ROOT to a ComfyUI checkout for node tests")
sys.path.insert(0, str(COMFY_ROOT))
comfy_args = importlib.import_module("comfy.cli_args").args
torch = importlib.import_module("torch")

if not torch.cuda.is_available():
    comfy_args.cpu = True
spec = importlib.util.spec_from_file_location(
    "unimate_node_test", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
extension_module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = extension_module
spec.loader.exec_module(extension_module)
asyncio.run(extension_module.comfy_entrypoint())
nodes = sys.modules[spec.name + ".nodes"]


class NodeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="unimate nodes ")
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.input = self.base / "input"
        self.output = self.base / "output"
        self.models = self.base / "models"
        for path in (self.input, self.output, self.models):
            path.mkdir()
        for name, value in (
            ("get_input_directory", self.input),
            ("get_output_directory", self.output),
        ):
            context = patch.object(nodes.folder_paths, name, return_value=str(value))
            context.start()
            self.addCleanup(context.stop)
        context = patch.dict(
            nodes.folder_paths.folder_names_and_paths,
            {"unimate": ([str(self.models)], {".unimate"})},
        )
        context.start()
        self.addCleanup(context.stop)

        # Inventory cache entries retain temporary directory mtimes too.
        context = patch.dict(nodes.folder_paths.filename_list_cache, {}, clear=True)
        context.start()
        self.addCleanup(context.stop)

    def fake_module(self, name, **functions):
        module = types.ModuleType(spec.name + ".unimate_pack." + name)
        module.__dict__.update(functions)
        return patch.dict(sys.modules, {module.__name__: module})

    def test_actual_comfy_extension_has_registered_valid_schemas(self):
        extension = asyncio.run(extension_module.comfy_entrypoint())
        classes = asyncio.run(extension.get_node_list())
        self.assertEqual(
            [cls.GET_SCHEMA().node_id for cls in classes],
            [
                "UniMateLoadRig",
                "UniMatePrepareRig",
                "UniMateModelLoader",
                "UniMateGenerateMotion",
                "UniMateExportGLB",
                "UniMateInbetweenMotion",
                "UniMateEditMotion",
                "UniMateLoadMotion",
                "UniMateSaveMotion",
                "UniMateExpandMotion",
                "UniMateExtractMotion",
                "UniMateGenerateBatch",
                "UniMateCombineRigs",
                "UniMateCanonicalAsset",
                "UniMateRigConditioning",
                "UniMateExportFBX",
                "UniMateRecoverSkeleton",
                "UniMatePreviewSkeleton",
                "UniMateFootLockMotion",
                "UniMateBuildDataset",
                "UniMateLoadDataset",
                "UniMateSaveDataset",
                "UniMateDatasetStatistics",
                "UniMateLoadStatistics",
                "UniMateSaveStatistics",
                "UniMateSplitDataset",
                "UniMatePlanSampling",
                "UniMateBuildTextCache",
                "UniMateTrainingJob",
                "UniMateTrain",
                "UniMateLoadTrainingCheckpoint",
                "UniMateSaveTrainingCheckpoint",
                "UniMateExportInferenceWeights",
                "UniMateLoadInferenceWeights",
                "UniMateCollateTrainingSamples",
                "UniMatePrepareTrainingSample",
            ],
        )
        for cls in classes:
            self.assertEqual(cls.GET_SCHEMA().category, "3D/UniMate")
        generate = nodes.UniMateGenerateMotion.GET_NODE_INFO_V1()
        self.assertEqual(
            generate["input"]["required"]["normalization"][1]["options"],
            ["objaverse", "mixamo", "truebones"],
        )
        self.assertTrue(nodes.UniMateExportGLB.OUTPUT_NODE)

    def test_dataset_selection_nodes_preserve_source_and_return_portable_plan(self):
        from test_dataset_selection import mixed_dataset
        from unimate_pack.dataset_selection import split_dataset, sampling_plan
        dataset = mixed_dataset()
        expected, report = split_dataset(dataset, .4, 17, '{}')
        result = nodes.UniMateSplitDataset.execute(dataset, .4, 17, '{}')
        self.assertEqual(result.result[0], expected)
        self.assertEqual(json.loads(result.result[1]), report)
        actual = nodes.UniMatePlanSampling.execute(expected, .5, True, .25, 3)
        self.assertEqual(actual.result[0], sampling_plan(expected, alpha=.5, dataset_alpha=.25, epoch=3))
        self.assertEqual(nodes.UniMatePlanSampling.GET_SCHEMA().outputs[0].io_type, 'UNIMATE_SAMPLING')

    def test_dataset_nodes_build_compute_save_load_and_declare_inputs(self):
        from test_foot_lock import portable_legged_motion
        from unimate_pack.dataset_io import load_dataset
        from unimate_pack.statistics_io import load_statistics
        rig, motion, _ = portable_legged_motion()
        result = nodes.UniMateBuildDataset.execute([rig], [motion], ['[]'], ['objaverse'])
        dataset = result.result[0]
        statistics = nodes.UniMateDatasetStatistics.execute(dataset, True, False, False).result[0]
        for save, load, value, read in (
            (nodes.UniMateSaveDataset, nodes.UniMateLoadDataset, dataset, load_dataset),
            (nodes.UniMateSaveStatistics, nodes.UniMateLoadStatistics, statistics, load_statistics),
        ):
            saved = save.execute(value, 'dataset test/left')
            descriptor = saved.ui['files'][0]
            path = self.output / descriptor['subfolder'] / descriptor['filename']
            self.assertEqual(read(path.read_bytes()), value)
            target = self.input / path.name
            target.write_bytes(path.read_bytes())
            self.assertEqual(load.execute(target.name).result[0], value)
            self.assertEqual(load.cloud_offload_assets({'archive': target.name}),
                             [{'category': '__input__', 'filename': target.name}])
            self.assertEqual(len(load.fingerprint_inputs(target.name)), 64)
        with self.assertRaises(ValueError):
            nodes.UniMateBuildDataset.execute([rig], [motion], ['[]', '[]'], ['objaverse'])

    def test_dataset_save_rejects_escape_and_cancellation_leaves_no_artifact(self):
        from test_dataset_contracts import dataset_parts
        from unimate_pack.dataset_contracts import make_dataset
        from comfy import model_management
        manifest, files = dataset_parts()
        dataset = make_dataset(manifest, files)
        with self.assertRaises(ValueError):
            nodes.UniMateSaveDataset.execute(dataset, '../outside')
        with patch.object(model_management, 'throw_exception_if_processing_interrupted',
                          side_effect=InterruptedError('cancelled')):
            with self.assertRaises(InterruptedError):
                nodes.UniMateSaveDataset.execute(dataset, 'cancelled/dataset')
        self.assertFalse(list(self.output.rglob('*.*')))

    def test_dataset_save_cancellation_after_stage_write_cleans_stage(self):
        from test_dataset_contracts import dataset_parts
        from unimate_pack.dataset_contracts import make_dataset
        from comfy import model_management
        dataset = make_dataset(*dataset_parts())
        calls = 0

        def cancel():
            nonlocal calls
            calls += 1
            if calls == 3:
                stages = list(self.output.rglob('.unimate-numeric-*'))
                self.assertEqual(len(stages), 1)
                self.assertEqual(stages[0].read_bytes(), b'staged archive')
                raise InterruptedError('cancelled before publication')

        with self.fake_module('dataset_io', dump_dataset=lambda value, cancel: b'staged archive'):
            with patch.object(model_management, 'throw_exception_if_processing_interrupted', side_effect=cancel):
                with self.assertRaises(InterruptedError):
                    nodes.UniMateSaveDataset.execute(dataset, 'staged/dataset')
        self.assertFalse(list(self.output.rglob('*.*')))

    def test_foot_lock_node_forwards_names_and_returns_portable_motion_and_report(self):
        corrected, report = {"rig_id": "a" * 64}, {"segments": []}
        calls = []
        def lock(rig, motion, joint_names, **kwargs):
            calls.append(joint_names)
            self.assertTrue(callable(kwargs["check_cancel"]))
            return corrected, report
        with self.fake_module("foot_lock", lock_motion=lock):
            result = nodes.UniMateFootLockMotion.execute({}, {}, "LeftToe, RightToe")
        self.assertEqual(result.result, (corrected, json.dumps(report, allow_nan=False)))
        self.assertEqual(calls, ["LeftToe, RightToe"])

    def test_extract_node_forwards_selected_clip_and_preserves_identity(self):
        with self.fake_module("source_motion", extract_motion=lambda rig, clip_index, **kwargs: {"rig_id": rig["rig_id"], "clip_index": clip_index}):
            output = nodes.UniMateExtractMotion.execute({"rig_id": "a" * 64}, 2)
            self.assertEqual(output.result[0], {"rig_id": "a" * 64, "clip_index": 2})

    def test_canonical_node_uses_existing_portable_asset_type(self):
        asset = {"glb": b"fixture"}
        with self.fake_module("canonical_asset", canonical_asset=lambda rig: asset):
            self.assertEqual(nodes.UniMateCanonicalAsset.execute({}).result, (asset,))
        self.assertEqual(nodes.UniMateCanonicalAsset.GET_NODE_INFO_V1()["output"], ["UNIMATE_ASSET"])

    def test_fbx_output_is_retrievable_with_matching_provenance(self):
        rig = {"rig_id": "rig", "asset": {"sha256": "source"}}
        motion = {"rig_id": "rig", "features": b"features", "fps": 30, "metadata": {}}
        with self.export_context(), self.fake_module("blender", export_fbx=lambda *args: b"FBX"):
            output = nodes.UniMateExportFBX.execute(rig, motion)
        files = output.ui["files"]
        self.assertEqual(len(files), 2)
        manifest = json.loads((self.output / files[1]["subfolder"] / files[1]["filename"]).read_text())
        self.assertEqual(manifest["fbx_sha256"], hashlib.sha256(b"FBX").hexdigest())

    def test_batch_node_emits_typed_list_and_preserves_prompt_order(self):
        calls = []
        def batch(*args):
            calls.append(args)
            return ([{"seed": i} for i in range(4)], [args[1][0]] * 4)
        with self.fake_module("batch", generate_batch_with_rigs=batch):
            output = nodes.UniMateGenerateBatch.execute([{}], [{"rig_id": "rig"}], ['["walk", "sit"]'], [2], [0], [3], ["objaverse"])
        self.assertEqual(output.result[0], [{"seed": i} for i in range(4)])
        self.assertEqual(output.result[1], [{"rig_id": "rig"}] * 4)
        self.assertEqual(calls[0][1:4], ([{"rig_id": "rig"}], ["walk", "sit"], 2))
        self.assertEqual(nodes.UniMateGenerateBatch.GET_NODE_INFO_V1()["output_is_list"], [True, True])
        self.assertTrue(nodes.UniMateGenerateBatch.GET_SCHEMA().is_input_list)

    def test_batch_consumes_multiple_rigs_once_and_rejects_mapped_controls(self):
        calls = []
        rigs = [{"rig_id": "a"}, {"rig_id": "b"}]
        def batch(*args):
            calls.append(args)
            return ([{"rig_id": "a"}, {"rig_id": "b"}], rigs)
        with self.fake_module("batch", generate_batch_with_rigs=batch):
            result = nodes.UniMateGenerateBatch.execute([{}], rigs, ['["walk"]'], [1], [3], [3], ["objaverse"])
            self.assertEqual(calls[0][1], rigs)
            self.assertEqual(result.result[1], rigs)
            with self.assertRaisesRegex(ValueError, 'one'):
                nodes.UniMateGenerateBatch.execute([{}, {}], rigs, ['["walk"]'], [1], [3], [3], ["objaverse"])
            self.assertEqual(len(calls), 1)

    def test_rig_collection_validates_and_preserves_input_order(self):
        rigs = [{"rig_id": "a"}, {"rig_id": "b"}, {"rig_id": "c"}]
        checked = []
        with self.fake_module("contracts", validate_rig=lambda rig: checked.append(rig)):
            result = nodes.UniMateCombineRigs.execute(rigs[:2], rigs[2:])
        self.assertEqual(result.result[0], rigs)
        self.assertEqual(checked, rigs)
        self.assertTrue(nodes.UniMateCombineRigs.GET_SCHEMA().is_input_list)
        self.assertEqual(nodes.UniMateCombineRigs.GET_NODE_INFO_V1()["output_is_list"], [True])

    def test_expansion_node_parses_prompt_array_without_reordering(self):
        calls = []
        def expand(*args, **kwargs):
            calls.append(args)
            return {"prompts": args[2]}
        with self.fake_module("expansion", expand_motion=expand):
            output = nodes.UniMateExpandMotion.execute({}, {}, '["stand", "walk"]', 0, 3, 10)
            self.assertEqual(output.result[0]["prompts"], ["stand", "walk"])
            with self.assertRaises(ValueError):
                nodes.UniMateExpandMotion.execute({}, {}, '{broken}', 0, 3, 10)

    def test_motion_archive_nodes_round_trip_and_confine_paths(self):
        import numpy as np
        from unimate_node_test.unimate_pack.contracts import make_motion, encode_arrays
        from unimate_node_test.unimate_pack.motion_io import load_motion
        motion = make_motion("a" * 64, encode_arrays(features=np.zeros((110, 5, 12), np.float32)), {})
        saved = nodes.UniMateSaveMotion.execute(motion, "motions/walk")
        item = saved.ui["files"][0]
        path = self.output / item["subfolder"] / item["filename"]
        self.assertEqual(load_motion(path.read_bytes()), motion)
        (self.input / "walk.npz").write_bytes(path.read_bytes())
        loaded = nodes.UniMateLoadMotion.execute("walk.npz")
        self.assertEqual(loaded.result[0], motion)
        self.assertEqual(nodes.UniMateLoadMotion.cloud_offload_assets({"archive": "walk.npz"}),
                         [{"category": "__input__", "filename": "walk.npz"}])
        for invalid in ("../walk.npz", "walk.glb", "missing.npz"):
            with self.assertRaises((ValueError, OSError)):
                nodes.UniMateLoadMotion.execute(invalid)

    def test_constrained_nodes_forward_reference_and_selection(self):
        calls = []
        def generate(*args, **kwargs):
            calls.append(kwargs)
            return {"rig_id": "a" * 64}
        with self.fake_module("inference", generate_motion=generate):
            for node, mode in ((nodes.UniMateInbetweenMotion, "inbetween"),
                               (nodes.UniMateEditMotion, "edit")):
                result = node.execute({}, {"rig_id": "a" * 64}, {"clip": 1}, "walk", 0, 3, "0,-1")
                self.assertEqual(result.result[0]["rig_id"], "a" * 64)
                self.assertEqual(calls[-1]["constraint_mode"], mode)
                self.assertEqual(calls[-1]["reference"], {"clip": 1})
                self.assertEqual(calls[-1]["selection"], "0,-1")

    def test_inputs_reject_escape_missing_and_wrong_extensions(self):
        for filename in (
            "../outside.glb",
            "..\\outside.glb",
            "/outside.glb",
            "C:\\outside.glb",
            "wrong.fbx",
            "missing.glb",
        ):
            with self.subTest(filename=filename):
                self.assertIsInstance(
                    nodes.UniMateLoadRig.validate_inputs(filename), str
                )
                with self.assertRaises((ValueError, FileNotFoundError)):
                    nodes.UniMateLoadRig.execute(filename)
        for filename in ("../outside.unimate", "wrong.zip", "missing.unimate"):
            with self.assertRaises((ValueError, FileNotFoundError)):
                nodes.UniMateModelLoader.execute(filename)

    def test_file_fingerprint_tracks_content_and_cloud_declaration(self):
        name = "character with spaces.glb"
        path = self.input / name
        path.write_bytes(b"first")
        first = nodes.UniMateLoadRig.fingerprint_inputs(name)
        path.write_bytes(b"second")
        self.assertNotEqual(nodes.UniMateLoadRig.fingerprint_inputs(name), first)
        self.assertEqual(
            nodes.UniMateLoadRig.cloud_offload_assets({"asset": name}),
            [{"category": "__input__", "filename": name}],
        )
        bundle = self.models / "official.unimate"
        bundle.write_bytes(b"model-one")
        first = nodes.UniMateModelLoader.fingerprint_inputs(bundle.name)
        bundle.write_bytes(b"model-two")
        self.assertNotEqual(
            nodes.UniMateModelLoader.fingerprint_inputs(bundle.name), first
        )

    def test_model_cloud_declaration_uses_exact_category_with_filename_collision(self):
        checkpoints = self.base / "checkpoints"
        checkpoints.mkdir()
        name = "official.unimate"
        (checkpoints / name).write_bytes(b"unrelated checkpoint")
        (self.models / name).write_bytes(b"selected UniMate bundle")
        with (
            patch.dict(
                nodes.folder_paths.folder_names_and_paths,
                {"checkpoints": ([str(checkpoints)], {".unimate"})},
            ),
            self.fake_module(
                "inference",
                load_model_bundle=lambda path: {"loaded": path.read_bytes()},
            ),
            self.fake_module("contracts", validate_model=lambda value: None),
        ):
            self.assertEqual(
                nodes.UniMateModelLoader.cloud_offload_assets({"bundle": name}),
                [{"category": "unimate", "filename": name}],
            )
            self.assertEqual(
                nodes.UniMateModelLoader.execute(name).result[0]["loaded"],
                b"selected UniMate bundle",
            )
            for invalid in ("../official.unimate", "missing.unimate", ["3", 0]):
                with self.assertRaises((ValueError, FileNotFoundError)):
                    nodes.UniMateModelLoader.cloud_offload_assets({"bundle": invalid})

    def test_load_asset_is_bytes_not_local_path(self):
        (self.input / "rig.glb").write_bytes(b"glb contents")
        asset = {
            "schema": "unimate.asset.v1",
            "glb": b"glb contents",
            "name": "rig.glb",
        }
        with (
            self.fake_module(
                "contracts",
                make_asset=lambda glb, name: {**asset, "glb": glb, "name": name},
            ),
            self.fake_module("assets", validate_glb=lambda glb: {}),
        ):
            self.assertEqual(nodes.UniMateLoadRig.execute("rig.glb").result, (asset,))

    def test_backend_delegation_and_normalization(self):
        model, rig, motion = {"schema": "model"}, {"rig_id": "same"}, {"rig_id": "same"}
        bundle = self.models / "selected.unimate"
        bundle.write_bytes(b"bundle")
        calls = []
        with (
            self.fake_module(
                "inference",
                load_model_bundle=lambda path: calls.append(path) or model,
                generate_motion=lambda *args, **kwargs: calls.append((args, kwargs))
                or motion,
            ),
            self.fake_module(
                "blender", prepare_rig=lambda *args: calls.append(args) or rig
            ),
            self.fake_module(
                "contracts",
                validate_model=lambda value: None,
                validate_rig=lambda value: None,
                validate_asset=lambda value: None,
                validate_motion=lambda value, rig_id=None: None,
            ),
        ):
            self.assertEqual(
                nodes.UniMateModelLoader.execute(bundle.name).result, (model,)
            )
            self.assertEqual(calls[-1], bundle.resolve())
            self.assertEqual(nodes.UniMatePrepareRig.execute({}, "+Z").result, (rig,))
            self.assertEqual(calls[-1], ({}, "+Z", "", ""))
            self.assertEqual(
                nodes.UniMateGenerateMotion.execute(
                    model, rig, "walk", 123, 3.0, "mixamo"
                ).result,
                (motion,),
            )
            self.assertEqual(
                calls[-1], ((model, rig, "walk", 123, 3.0), {"normalization": "mixamo"})
            )

    def test_invalid_sampling_options_and_joint_pair_fail_before_backend(self):
        with (
            self.fake_module(
                "inference",
                generate_motion=lambda *args, **kwargs: self.fail("backend called"),
            ),
            self.fake_module(
                "blender", prepare_rig=lambda *args: self.fail("backend called")
            ),
            self.fake_module(
                "contracts",
                validate_model=lambda value: None,
                validate_rig=lambda value: None,
                validate_asset=lambda value: None,
                validate_motion=lambda value, rig_id=None: None,
            ),
        ):
            for changes in (
                {"seed": -1},
                {"seed": True},
                {"seed": 2**64},
                {"guidance": float("nan")},
                {"guidance": 11},
                {"normalization": "unknown"},
            ):
                inputs = dict(
                    model={},
                    rig={"rig_id": "same"},
                    prompt="walk",
                    seed=0,
                    guidance=3.0,
                )
                inputs.update(changes)
                with self.assertRaises(ValueError):
                    nodes.UniMateGenerateMotion.execute(**inputs)
            for names in (("", ""), ("left", ""), ("same", "same")):
                with self.assertRaises(ValueError):
                    nodes.UniMatePrepareRig.execute({}, "joint_pair", *names)

    def test_cancelled_backend_does_not_publish_output(self):
        def cancelled(*args, **kwargs):
            raise InterruptedError("cancelled")

        rig = {"rig_id": "same"}
        with (
            self.fake_module("inference", generate_motion=cancelled),
            self.fake_module("blender", export_glb=cancelled),
            self.fake_module(
                "contracts",
                validate_model=lambda value: None,
                validate_rig=lambda value: None,
                validate_motion=lambda value, rig_id=None: None,
            ),
        ):
            with self.assertRaises(InterruptedError):
                nodes.UniMateGenerateMotion.execute({}, rig, "walk", 0, 3.0)
            with self.assertRaises(InterruptedError):
                nodes.UniMateExportGLB.execute(rig, {"rig_id": "same"}, "animation")
        self.assertFalse(list(self.output.rglob("*")))

    def test_input_symlink_escape_is_rejected(self):
        outside = self.base / "outside.glb"
        outside.write_bytes(b"outside")
        try:
            (self.input / "escape.glb").symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        with self.assertRaises(ValueError):
            nodes.UniMateLoadRig.fingerprint_inputs("escape.glb")

    def test_model_symlink_escape_is_rejected(self):
        outside = self.base / "outside.unimate"
        outside.write_bytes(b"outside")
        try:
            (self.models / "escape.unimate").symlink_to(outside)
        except OSError as error:
            self.skipTest(f"Symlink creation unavailable: {error}")
        with self.assertRaisesRegex(ValueError, "outside"):
            nodes.UniMateModelLoader.fingerprint_inputs("escape.unimate")

    def export_context(self):
        def validate_motion(value, rig_id=None):
            expected = rig_id['rig_id'] if isinstance(rig_id, dict) else rig_id
            if expected is not None and value.get('rig_id') != expected:
                raise ValueError('Motion and prepared rig identities do not match')
        return self.fake_module(
            "contracts",
            validate_rig=lambda value: None,
            validate_motion=validate_motion,
        )

    def test_public_export_accepts_exact_legacy_archive_identity(self):
        from test_rig_portability import archive_for_platform, legacy_identity
        from unimate_pack.contracts import make_asset, make_rig, make_motion, encode_arrays
        from unimate_pack.rig_math import parse_glb, prepare_document
        from test_blender_math import synthetic_glb
        import numpy as np

        source = synthetic_glb(True)
        asset = make_asset(source, 'rig.glb')
        arrays, mapping = prepare_document(parse_glb(source)[0], '+Z')
        archive = archive_for_platform(arrays, 0)
        rig = make_rig(asset, archive, mapping)
        motion = make_motion(legacy_identity(asset, archive, mapping),
            encode_arrays(features=np.zeros((7, 7, 12), np.float32)), {})
        self.assertNotEqual(motion['rig_id'], rig['rig_id'])
        with self.fake_module('blender', export_glb=lambda *args: b'GLB', export_fbx=lambda *args: b'FBX'):
            for node in (nodes.UniMateExportGLB, nodes.UniMateExportFBX):
                result = node.execute(rig, motion, 'legacy/' + node.FORMAT)
                self.assertTrue(result.ui['files'])

    def test_export_unique_files_and_provenance_without_paths(self):
        rig = {"rig_id": "rig-identity", "asset": {"sha256": "source-digest"}}
        motion = {
            "rig_id": "rig-identity",
            "fps": 30,
            "features": b"features",
            "metadata": {"prompt": "walk", "seed": 42},
        }
        with (
            self.export_context(),
            self.fake_module("blender", export_glb=lambda *args: b"animated GLB"),
        ):
            results = [
                nodes.UniMateExportGLB.execute(
                    rig, motion, "with spaces/日本語/animation"
                ).ui
                for _ in range(2)
            ]
        self.assertNotEqual(
            results[0]["3d"][0]["subfolder"], results[1]["3d"][0]["subfolder"]
        )
        for ui in results:
            self.assertEqual(set(ui), {"3d", "files"})
            glb, provenance = ui["3d"][0], ui["files"][0]
            self.assertEqual(glb["type"], "output")
            self.assertEqual(
                (self.output / glb["subfolder"] / glb["filename"]).read_bytes(),
                b"animated GLB",
            )
            manifest_text = (
                self.output / provenance["subfolder"] / provenance["filename"]
            ).read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            self.assertEqual(
                manifest["glb_sha256"], hashlib.sha256(b"animated GLB").hexdigest()
            )
            self.assertEqual(manifest["generation"], motion["metadata"])
            self.assertNotIn(str(self.base), manifest_text)
        self.assertFalse(list(self.output.rglob("*.tmp")))

    def test_export_rejects_mismatched_motion_and_traversal_before_backend(self):
        rig = {"rig_id": "rig-a", "asset": {"sha256": "source"}}
        with (
            self.export_context(),
            self.fake_module(
                "blender", export_glb=lambda *args: self.fail("backend called")
            ),
        ):
            with self.assertRaisesRegex(ValueError, "rig"):
                nodes.UniMateExportGLB.execute(rig, {"rig_id": "rig-b"}, "animation")
            for prefix in (
                "../outside",
                "..\\outside",
                "/absolute",
                "C:\\absolute",
                "animation:stream",
            ):
                with self.assertRaises(ValueError):
                    nodes.UniMateExportGLB.execute(rig, {"rig_id": "rig-a"}, prefix)

    def test_failed_publication_leaves_no_partial_output(self):
        rig = {"rig_id": "rig-a", "asset": {"sha256": "source"}}
        motion = {"rig_id": "rig-a", "fps": 30, "features": b"features", "metadata": {}}
        with (
            self.export_context(),
            self.fake_module("blender", export_glb=lambda *args: b"GLB"),
            patch.object(
                nodes.os, "replace", side_effect=OSError("simulated full disk")
            ),
        ):
            with self.assertRaises(OSError):
                nodes.UniMateExportGLB.execute(rig, motion, "animation")
        self.assertFalse([path for path in self.output.rglob("*") if path.is_file()])

    def test_interruption_during_fsync_cleans_staged_pair_before_publication(self):
        from comfy import model_management

        self.addCleanup(model_management.interrupt_current_processing, False)
        model_management.interrupt_current_processing(False)
        rig = {"rig_id": "rig-a", "asset": {"sha256": "source"}}
        motion = {"rig_id": "rig-a", "fps": 30, "features": b"features", "metadata": {}}
        real_fsync = os.fsync

        def signal_during_fsync(descriptor):
            real_fsync(descriptor)
            model_management.interrupt_current_processing(True)

        with (
            self.export_context(),
            self.fake_module("blender", export_glb=lambda *args: b"GLB"),
            patch.object(nodes.os, "fsync", side_effect=signal_during_fsync),
            patch.object(nodes.os, "replace", wraps=os.replace) as publish,
        ):
            with self.assertRaises(model_management.InterruptProcessingException):
                nodes.UniMateExportGLB.execute(rig, motion, "animation")
            publish.assert_not_called()
        self.assertFalse(list(self.output.rglob("*")))


if __name__ == "__main__":
    unittest.main()
