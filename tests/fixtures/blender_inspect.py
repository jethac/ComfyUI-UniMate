"""Independent Blender playback probe for generated original synthetic fixtures."""

import bpy
import sys
import numpy as np

source, output = sys.argv[sys.argv.index("--") + 1 :]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.context.scene.render.fps = 30
bpy.ops.import_scene.gltf(filepath=source, bone_heuristic="BLENDER")
frames = []
for frame in range(60):
    bpy.context.scene.frame_set(frame)
    graph = bpy.context.evaluated_depsgraph_get()
    vertices = []
    for obj in sorted(bpy.context.scene.objects, key=lambda x: x.name):
        if obj.type != "MESH" or not any(m.type == "ARMATURE" for m in obj.modifiers):
            continue
        evaluated = obj.evaluated_get(graph)
        mesh = evaluated.to_mesh()
        try:
            for v in mesh.vertices:
                w = evaluated.matrix_world @ v.co
                vertices.append([w.x, w.z, -w.y])
        finally:
            evaluated.to_mesh_clear()
    frames.append(vertices)
np.savez(output, vertices=np.asarray(frames))
