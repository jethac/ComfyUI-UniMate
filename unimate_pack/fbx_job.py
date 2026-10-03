"""External Blender FBX conversion with per-frame skinning round-trip checks."""

from pathlib import Path
import sys

import bpy
import numpy as np
from mathutils.kdtree import KDTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from unimate_pack.blender_job import load, evaluate_vertices


def image_pixels():
    return [(tuple(image.size), np.asarray(image.pixels[:], dtype=np.float32))
            for image in bpy.data.images if image.users and image.source == "FILE"]


def prepare_base_color_textures(folder):
    """Fold glTF's constant base-color multiplier into an FBX-readable image."""
    for material in bpy.data.materials:
        if material.node_tree is None:
            continue
        tree = material.node_tree
        for shader in list(tree.nodes):
            if shader.type != "BSDF_PRINCIPLED":
                continue
            socket = shader.inputs["Base Color"]
            if not socket.is_linked or socket.links[0].from_node.type != "MIX":
                continue
            mix = socket.links[0].from_node
            if (mix.blend_type != "MULTIPLY" or mix.data_type != "RGBA"
                    or mix.inputs[0].is_linked or mix.inputs[0].default_value != 1
                    or not mix.inputs[6].is_linked or mix.inputs[7].is_linked):
                raise ValueError("Unsupported FBX base-color expression")
            texture = mix.inputs[6].links[0].from_node
            if texture.type != "TEX_IMAGE" or texture.image is None:
                raise ValueError("Unsupported FBX base-color texture")
            pixels = np.asarray(texture.image.pixels[:], dtype=np.float32).reshape(-1, 4)
            pixels[:, :3] *= np.asarray(mix.inputs[7].default_value[:3])
            image = bpy.data.images.new(f"UniMateBaseColor-{material.name}",
                width=texture.image.size[0], height=texture.image.size[1], alpha=True,
                float_buffer=True)
            image.pixels[:] = pixels.reshape(-1)
            image.filepath_raw = str(folder / f"texture-baked-{len(bpy.data.images)}.png")
            scene = bpy.context.scene
            scene.render.image_settings.file_format = "PNG"
            scene.render.image_settings.color_mode = "RGBA"
            scene.render.image_settings.color_depth = "16"
            scene.view_settings.view_transform = "Standard"
            scene.view_settings.look = "None"
            image.save_render(image.filepath_raw, scene=scene)
            # Reload the exact embedded pixels, and bound PNG quantization loss.
            stored = bpy.data.images.load(image.filepath_raw, check_existing=False)
            if not np.allclose(np.asarray(stored.pixels[:]).reshape(-1, 4), pixels,
                    atol=5e-5, rtol=0):
                raise ValueError("FBX base-color PNG changed effective texture colors")
            replacement = tree.nodes.new("ShaderNodeTexImage")
            replacement.image = stored
            replacement.interpolation = texture.interpolation
            replacement.extension = texture.extension
            if texture.inputs["Vector"].is_linked:
                tree.links.new(texture.inputs["Vector"].links[0].from_socket,
                    replacement.inputs["Vector"])
            tree.links.new(replacement.outputs["Color"], socket)
            tree.nodes.remove(mix)
            if not texture.outputs["Color"].is_linked and not texture.outputs["Alpha"].is_linked:
                tree.nodes.remove(texture)


def main():
    folder = Path(sys.argv[sys.argv.index("--") + 1])
    frames = int(sys.argv[sys.argv.index("--") + 2])
    armature = load(folder / "input.glb")
    names = set(armature.data.bones.keys())
    scene = bpy.context.scene
    prepare_base_color_textures(folder)
    expected_images = image_pixels()
    scene.frame_start, scene.frame_end = 0, frames - 1
    for frame in range(frames):
        scene.frame_set(frame)
        np.save(folder / f"frame-{frame}.npy", evaluate_vertices(), allow_pickle=False)
    for index, image in enumerate(bpy.data.images):
        if image.has_data and image.users and not Path(image.filepath_raw).is_file():
            image.filepath_raw = str(folder / f"texture-{index}.png")
            image.file_format = "PNG"
            image.save()
    bpy.ops.export_scene.fbx(filepath=str(folder / "output.fbx"),
        add_leaf_bones=False, bake_anim=True, bake_anim_use_all_actions=False,
        bake_anim_use_nla_strips=False, bake_anim_simplify_factor=0,
        path_mode="COPY", embed_textures=True, axis_forward="-Z", axis_up="Y")
    # Remove sidecar images so reimport must resolve the embedded FBX content.
    for path in folder.glob("texture-*.png"):
        path.unlink()
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.context.scene.render.fps = 30
    bpy.ops.import_scene.fbx(filepath=str(folder / "output.fbx"), anim_offset=0)
    armatures = [obj for obj in bpy.context.scene.objects if obj.type == "ARMATURE"]
    if len(armatures) != 1 or set(armatures[0].data.bones.keys()) != names:
        raise ValueError("FBX round trip changed skeleton identity")
    actual_images = image_pixels()
    if len(actual_images) != len(expected_images):
        raise ValueError(f"FBX round trip changed texture inventory: expected {len(expected_images)}, "
            f"actual {len(actual_images)}; images "
            f"{[(image.name, image.source, tuple(image.size)) for image in bpy.data.images]}")
    for size, pixels in expected_images:
        match = next((index for index, (other_size, other_pixels) in enumerate(actual_images)
            if size == other_size and pixels.shape == other_pixels.shape
            and np.allclose(pixels, other_pixels, atol=2e-6, rtol=0)), None)
        if match is None:
            raise ValueError("FBX round trip changed embedded texture pixels")
        actual_images.pop(match)
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
