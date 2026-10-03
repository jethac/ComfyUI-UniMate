"""External Blender FBX conversion with per-frame skinning round-trip checks."""

from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unimate_pack.blender_job import load, evaluate_vertices


def main():
    folder = Path(sys.argv[sys.argv.index("--") + 1])
    frames = int(sys.argv[sys.argv.index("--") + 2])
    armature = load(folder / "input.glb")
    names = set(armature.data.bones.keys())
    scene = bpy.context.scene
    scene.frame_start, scene.frame_end = 0, frames - 1
    for frame in range(frames):
        scene.frame_set(frame)
        np.save(folder / f"frame-{frame}.npy", evaluate_vertices(), allow_pickle=False)
    for index, image in enumerate(bpy.data.images):
        if image.has_data:
            image.filepath_raw = str(folder / f"texture-{index}.png")
            image.file_format = "PNG"
            image.save()
    bpy.ops.export_scene.fbx(filepath=str(folder / "output.fbx"),
        add_leaf_bones=False, bake_anim=True, bake_anim_use_all_actions=False,
        bake_anim_use_nla_strips=False, bake_anim_simplify_factor=0,
        path_mode="COPY", embed_textures=True, axis_forward="-Z", axis_up="Y")
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.fbx(filepath=str(folder / "output.fbx"), anim_offset=0)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1 or set(armatures[0].data.bones.keys()) != names:
        raise ValueError("FBX round trip changed skeleton identity")
    for frame in range(frames):
        bpy.context.scene.frame_set(frame)
        actual = evaluate_vertices()
        expected = np.load(folder / f"frame-{frame}.npy", allow_pickle=False)
        if actual.shape != expected.shape:
            raise ValueError("FBX round trip changed vertex inventory")
        for a, b in ((actual, expected), (expected, actual)):
            tree = KDTree(len(b))
            for index, vertex in enumerate(b):
                tree.insert(vertex, index)
            tree.balance()
            if any(tree.find(vertex)[2] > 2e-4 for vertex in a):
                raise ValueError(f"FBX round trip changed skinning at frame {frame}")


if __name__ == "__main__":
    main()
