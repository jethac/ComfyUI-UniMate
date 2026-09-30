"""Execute the pinned upstream extraction function for local reference tests."""

import ast
import sys
from pathlib import Path
import bpy
import numpy as np

source, upstream, output = sys.argv[sys.argv.index("--") + 1 :]
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.gltf(filepath=source, bone_heuristic="BLENDER")
armature = next(o for o in bpy.context.scene.objects if o.type == "ARMATURE")
path = Path(upstream) / "data_process/utils/blender_export.py"
module = ast.parse(path.read_text(encoding="utf-8"))
selected = [
    n
    for n in module.body
    if isinstance(n, ast.FunctionDef) and n.name == "extract_rest_pose"
]
reference_namespace = {"np": np}
exec(
    compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
    reference_namespace,
)
positions, rotations = reference_namespace["extract_rest_pose"](
    armature.data.bones, armature, apply_world=True
)
np.savez(
    output,
    names=np.array(list(armature.data.bones.keys())),
    parents=np.array(
        [
            -1 if b.parent is None else armature.data.bones.find(b.parent.name)
            for b in armature.data.bones
        ]
    ),
    rest_local_pos=positions,
    rest_local_rot=rotations,
    fps=np.array(30),
)
