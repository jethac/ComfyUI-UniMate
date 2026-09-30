"""Trusted external worker. Invoked only by blender.py with private job files."""

import json
from pathlib import Path
import sys
import numpy as np
import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unimate_pack.rig_math import (
    prepare_document,
    parse_glb,
    pack_glb,
    animate_document,
)
from unimate_pack.contracts import encode_arrays, decode_arrays


def load(path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.gltf(
        filepath=str(path), import_pack_images=True, bone_heuristic="BLENDER"
    )
    armatures = [o for o in bpy.context.scene.objects if o.type == "ARMATURE"]
    if len(armatures) != 1:
        raise ValueError("GLB must import as exactly one Blender armature")
    return armatures[0]


def rest_matrices(armature, names):
    # Same local extraction and world-root convention as pinned
    # blender_export.extract_rest_pose, including world scale normalization.
    missing = set(names) - set(armature.data.bones.keys())
    if missing:
        raise ValueError(f"Blender changed/missed joint identities: {sorted(missing)}")
    result = {}
    conversion = Matrix.Rotation(-np.pi / 2, 4, "X")

    def resolve(bone):
        if bone.name in result:
            return result[bone.name]
        if bone.parent:
            local = bone.parent.matrix_local.inverted() @ bone.matrix_local
            m = (
                Matrix.Translation(local.to_translation())
                @ local.to_quaternion().to_matrix().to_4x4()
            )
            world = resolve(bone.parent) @ m
        else:
            m = armature.matrix_world @ bone.matrix_local
            scale = armature.matrix_world.to_scale()
            p = m.to_translation()
            for i in range(3):
                p[i] /= scale[i]
            world = conversion @ (
                Matrix.Translation(p) @ m.to_quaternion().to_matrix().to_4x4()
            )
        result[bone.name] = world
        return world

    for name in names:
        resolve(armature.data.bones[name])
    return {name: np.array(result[name]).tolist() for name in names}


def evaluate_vertices():
    graph = bpy.context.evaluated_depsgraph_get()
    result = []
    for obj in sorted(bpy.context.scene.objects, key=lambda o: o.name):
        if obj.type != "MESH" or not any(m.type == "ARMATURE" for m in obj.modifiers):
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            result.extend([list(evaluated.matrix_world @ v.co) for v in mesh.vertices])
        finally:
            evaluated.to_mesh_clear()
    a = np.asarray(result)
    if not a.size or not np.isfinite(a).all():
        raise ValueError("Blender evaluated invalid skinned geometry")
    return a


def main():
    job = Path(sys.argv[sys.argv.index("--") + 1]).resolve()
    folder = job.parent
    request = json.loads(job.read_text(encoding="utf-8"))
    source = (folder / "input.glb").read_bytes()
    doc, _ = parse_glb(source)
    # Ignore source clips BEFORE import. Clearing animation_data afterwards
    # retains evaluated object TRS, including animated non-joint ancestors.
    rest_doc = dict(doc)
    rest_doc.pop("animations", None)
    _, original_binary = parse_glb(source)
    (folder / "rest.glb").write_bytes(pack_glb(rest_doc, original_binary))
    armature = load(folder / "rest.glb")
    for obj in bpy.context.scene.objects:
        if obj.animation_data:
            obj.animation_data_clear()
    for pb in armature.pose.bones:
        pb.matrix_basis = Matrix.Identity(4)
    bpy.context.view_layer.update()
    rest_vertices = evaluate_vertices()
    if request["operation"] == "prepare":
        names = [
            doc["nodes"][j].get("name", f"joint_{j}") for j in doc["skins"][0]["joints"]
        ]
        cond, mapping = prepare_document(
            doc,
            request["facing"],
            request["left_joint"],
            request["right_joint"],
            request["name"],
            rest_matrices(armature, names),
        )
        (folder / "conditioning.npz").write_bytes(encode_arrays(**cond))
        report = {
            "mapping": mapping,
            "blender": bpy.app.version_string,
            "rest_vertex_count": len(rest_vertices),
        }
    elif request["operation"] == "export":
        cond = decode_arrays((folder / "conditioning.npz").read_bytes())
        features = decode_arrays((folder / "features.npz").read_bytes())["features"]
        output = animate_document(source, cond, request["mapping"], features)
        (folder / "output.glb").write_bytes(output)
        armature = load(folder / "output.glb")
        scene = bpy.context.scene
        for frame in range(60):
            scene.frame_set(frame)
            evaluate_vertices()
        # Disable the clip and verify the original binding still evaluates identically.
        for obj in bpy.context.scene.objects:
            if obj.animation_data:
                obj.animation_data_clear()
        for pb in armature.pose.bones:
            pb.matrix_basis = Matrix.Identity(4)
        bpy.context.view_layer.update()
        after = evaluate_vertices()
        if after.shape != rest_vertices.shape or not np.allclose(
            after, rest_vertices, atol=2e-5, rtol=2e-5
        ):
            raise ValueError("Export changed Blender rest-pose geometry")
        report = {
            "blender": bpy.app.version_string,
            "frames_verified": 60,
            "rest_max_error": float(np.max(np.abs(after - rest_vertices))),
        }
    else:
        raise ValueError("Unknown Blender operation")
    (folder / "report.json").write_text(
        json.dumps(report, allow_nan=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
