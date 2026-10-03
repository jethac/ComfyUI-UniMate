"""Orthographic skeleton frames with fixed bounds and bounded output allocation."""

import numpy as np
from PIL import Image, ImageDraw

from .contracts import MAX_ARRAY_BYTES, decode_arrays, validate_skeleton


def render_skeleton(skeleton, projection='front', resolution=256, *, check_cancel=lambda: None):
    check_cancel()
    validate_skeleton(skeleton)
    axes = {'front': (0, 1), 'side': (2, 1), 'top': (0, 2)}
    if projection not in axes:
        raise ValueError('Skeleton projection must be front, side or top')
    arrays = decode_arrays(skeleton['arrays'])
    positions, parents = arrays['positions'], arrays['parents']
    if (type(resolution) is not int or not 64 <= resolution <= 1024
            or len(positions) * resolution * resolution * 3 * 4 > MAX_ARRAY_BYTES):
        raise ValueError('Skeleton image allocation exceeds budget or resolution bounds')
    points = positions[..., list(axes[projection])].astype(np.float64)
    low, high = points.min(axis=(0, 1)), points.max(axis=(0, 1))
    center = low + (high - low) / 2
    scale = (resolution - 16) / max(float((high - low).max()), 1e-6)
    points = (points - center) * scale
    points[..., 1] *= -1
    points += resolution / 2
    images = np.empty((len(points), resolution, resolution, 3), dtype=np.float32)
    for frame, joints in enumerate(points):
        check_cancel()
        image = Image.new('RGB', (resolution, resolution), (20, 24, 30))
        draw = ImageDraw.Draw(image)
        for joint, parent in enumerate(parents[1:], 1):
            draw.line([tuple(joints[parent]), tuple(joints[joint])], fill=(80, 180, 255), width=2)
        for joint, (x, y) in enumerate(joints):
            color = (255, 160, 50) if joint == 0 else (235, 240, 245)
            draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
        images[frame] = np.asarray(image, dtype=np.float32) / 255
    check_cancel()
    return images
