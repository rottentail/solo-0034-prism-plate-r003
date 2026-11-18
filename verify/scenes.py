"""Deterministic scenes used by the verification journeys.

Two families:

* ``selection`` - a 128x128 plate whose successive revisions move a local
  even-odd hole and change the opacity of a screen-blended transparent glaze.
  Interior probes make stale geometry and stale/absent transparent compositing
  separately visible in a real PNG.
* ``slow`` / ``gate`` / ``inflation`` - budget-valid but deliberately expensive
  scenes used only to control scheduling (occupancy, queue position, commit
  duration) without touching the server.

All documents pass the public admission rules (1..16 layers, 1..32 shapes,
<=256 contour verbs, <=512 canvas, <=160M segment*canvas*4 estimate).
"""
from __future__ import annotations

import math

SELECTION_SIZE = 128
SELECTION_REVISIONS = [
    {"hole": (20, 20, 44, 44), "glaze": 0.60},
    {"hole": (72, 72, 96, 96), "glaze": 0.25},
    {"hole": (52, 52, 68, 68), "glaze": 0.40},
    {"hole": (16, 112, 28, 124), "glaze": 0.15},
]

# Whole-pixel interior probes.  None touches a contour edge, so all four
# coverage samples of the pixel share the same state.
PROBES = {
    "background": (4, 4),
    "ink_outside_glaze": (100, 20),
    "overlay": (16, 96),          # ink under the transparent screen glaze, never a hole
    "hole_r1": (32, 32),          # hole in r1; ink under glaze in r2..r4
    "hole_r2": (84, 84),          # hole in r2; plain ink in r1/r3/r4
    "hole_r3": (60, 60),          # hole in r3; ink under glaze in r1/r2/r4
    "hole_r4": (22, 118),         # hole in r4; plain ink in r1..r3
}

HOLE_PROBES = {
    0: "hole_r1",
    1: "hole_r2",
    2: "hole_r3",
    3: "hole_r4",
}

BACKGROUND = {"r": 0.05, "g": 0.08, "b": 0.12, "a": 1.0}
INK = {"r": 0.85, "g": 0.35, "b": 0.10, "a": 1.0}
GLAZE = {"r": 0.10, "g": 0.70, "b": 0.85, "a": 0.80}


def rect(x0: float, y0: float, x1: float, y1: float) -> list[dict]:
    return [
        {"kind": "M", "points": [{"x": x0, "y": y0}]},
        {"kind": "L", "points": [{"x": x1, "y": y0}]},
        {"kind": "L", "points": [{"x": x1, "y": y1}]},
        {"kind": "L", "points": [{"x": x0, "y": y1}]},
        {"kind": "Z"},
    ]


def selection_document(index: int, glaze_override: float | None = None) -> dict:
    spec = SELECTION_REVISIONS[index]
    glaze = spec["glaze"] if glaze_override is None else glaze_override
    return {
        "width": SELECTION_SIZE,
        "height": SELECTION_SIZE,
        "background": dict(BACKGROUND),
        "layers": [
            {
                "id": "ink",
                "opacity": 1.0,
                "blend": "normal",
                "shapes": [
                    {
                        "id": "ink-plate",
                        "fill_rule": "evenodd",
                        "path": rect(8, 8, 120, 120) + rect(*spec["hole"]),
                        "transform": [1, 0, 0, 1, 0, 0],
                        "brush": {"kind": "solid", "color": dict(INK)},
                    }
                ],
            },
            {
                "id": "glaze",
                "opacity": glaze,
                "blend": "screen",
                "shapes": [
                    {
                        "id": "glaze-field",
                        "fill_rule": "nonzero",
                        "path": rect(8, 8, 64, 128),
                        "transform": [1, 0, 0, 1, 0, 0],
                        "brush": {"kind": "solid", "color": dict(GLAZE)},
                    }
                ],
            },
        ],
    }


def selection_without_glaze(index: int) -> dict:
    return selection_document(index, glaze_override=0.0)


def slow_document(size: int = 512, vertices: int = 140, color=None) -> dict:
    """A single large contour; ~0.24 s of raster work on the verification host."""
    colour = color or {"r": 1.0, "g": 0.0, "b": 0.0, "a": 1.0}
    cx = cy = size / 2.0
    radius = size * 0.42
    path = [{"kind": "M", "points": [{"x": cx + radius, "y": cy}]}]
    for i in range(1, vertices):
        angle = i * 2.0 * math.pi / vertices
        path.append({"kind": "L", "points": [
            {"x": cx + radius * math.cos(angle), "y": cy + radius * math.sin(angle)}
        ]})
    path.append({"kind": "Z"})
    return {
        "width": size,
        "height": size,
        "background": {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0},
        "layers": [{
            "id": "slow",
            "opacity": 1.0,
            "blend": "normal",
            "shapes": [{
                "id": "slow-shape",
                "fill_rule": "nonzero",
                "path": path,
                "transform": [1, 0, 0, 1, 0, 0],
                "brush": {"kind": "solid", "color": colour},
            }],
        }],
    }


def gate_document(size: int = 512, shapes: int = 8) -> dict:
    """Occupies both render slots for ~1.4 s when two copies run in parallel."""
    compiled = []
    for k in range(shapes):
        compiled.append({
            "id": f"gate-{k}",
            "fill_rule": "evenodd",
            "path": rect(4 + k, 4, size - 4, size - 4),
            "transform": [1, 0.02 * k, 0, 1, 0, 0],
            "brush": {"kind": "solid",
                      "color": {"r": 0.2 + k / 40.0, "g": 0.4, "b": 0.7, "a": 0.9}},
        })
    return {
        "width": size,
        "height": size,
        "background": {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0},
        "layers": [{"id": "gate", "opacity": 1.0, "blend": "normal", "shapes": compiled}],
    }


def inflation_document(tag: str, shapes: int = 26, size: int = 8) -> dict:
    """Large vector artifact (~3.2 MB SVG) used to widen the snapshot commit.

    This is verification scaffolding: it makes the persistence step expensive
    enough to observe externally.  It does not change server logic.
    """
    ops = [{"kind": "M", "points": [{"x": 1, "y": 1}]}]
    for i in range(253):
        ops.append({"kind": "C", "points": [
            {"x": 1 + (i % 5) * 0.3, "y": 1 + (i % 7) * 0.2},
            {"x": ((i * 3) % 7) * 0.1 + 0.5, "y": ((i * 5) % 7) * 0.1 + 0.5},
            {"x": 1.2 + (i % 3) * 0.1, "y": 1.3 + ((i * 2) % 3) * 0.1},
        ]})
    ops.append({"kind": "Z"})
    compiled = [{
        "id": f"{tag}-shape-{k}",
        "fill_rule": "nonzero",
        "path": ops,
        "transform": [1, 0, 0, 1, 0, 0],
        "brush": {"kind": "solid",
                  "color": {"r": (k + 1) / 27.0, "g": 0.6, "b": 0.5, "a": 1.0}},
    } for k in range(shapes)]
    return {
        "width": size,
        "height": size,
        "background": {"r": 0.0, "g": 0.0, "b": 0.0, "a": 1.0},
        "layers": [{"id": f"{tag}-layer", "opacity": 1.0, "blend": "normal", "shapes": compiled}],
    }


def variant_revision(index: int) -> dict:
    """A slow revision with distinct colour, so every commit creates a new artifact.

    Index 0..124 maps to a unique (r, g, b) triple in 0..1.
    """
    colour = {"r": (index % 5) / 5.0 + 0.1,
              "g": ((index // 5) % 5) / 5.0,
              "b": ((index // 25) % 5) / 5.0,
              "a": 1.0}
    return slow_document(512, 140, colour)
