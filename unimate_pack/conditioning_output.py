"""Portable prepared topology conditioning with canonical rig identity."""

import copy
import hashlib

from .contracts import validate_rig


def extract_conditioning(rig):
    validate_rig(rig)
    return {"schema": "unimate.conditioning.v1", "rig_id": rig["rig_id"],
            "arrays": rig["conditioning"],
            "sha256": hashlib.sha256(rig["conditioning"]).hexdigest(),
            "mapping": copy.deepcopy(rig["mapping"])}
