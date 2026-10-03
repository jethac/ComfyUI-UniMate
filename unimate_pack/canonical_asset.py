"""Preserve the original asset under its prepared coordinate transform."""

import numpy as np

from .contracts import make_asset, validate_rig
from .rig_math import parse_glb, pack_glb


def canonical_asset(rig):
    validate_rig(rig)
    document, binary = parse_glb(rig["asset"]["glb"])
    scene = document["scenes"][document.get("scene", 0)]
    transform = np.asarray(rig["mapping"]["source_to_canonical"])
    index = len(document["nodes"])
    document["nodes"].append({"name": "UniMateCanonicalBasis",
        "matrix": transform.T.reshape(-1).tolist(), "children": scene["nodes"][:]})
    scene["nodes"] = [index]
    return make_asset(pack_glb(document, binary), "canonical.glb")
