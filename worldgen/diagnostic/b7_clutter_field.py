# Copyright 2026 Mert Güler
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SC7 Clutter field, variants B7a (45 blocks) and B7b (90 blocks).

All lengths of the placement are integers in units of 0.05 m (the raster cell), so block edges lie on cell
boundaries. B7a's blocks are the first 45 blocks of B7b.
"""

from __future__ import annotations

import functools
import math

import numpy as np
from scipy import ndimage

from ..checks import Analysis, no_symmetry
from ..geometry import box, walled_rectangle
from ..world import (
    CheckResult,
    Dimension,
    Expected,
    ExtraCheck,
    Primitive,
    Role,
    Spawn,
    World,
)

UNIT = 0.05
HALF = 300  # interior half-size, 15.0 m
WALL_GAP = 20  # 1.0 m
BLOCK_GAP = 20  # 1.0 m
SPAWN_EXCLUSION = 30  # 1.5 m
BLOCKS = 90
ATTEMPTS = 20_000
HEIGHT = 1.2
SEEDS = range(7100, 7900)
VARIANTS = {"SC7a": (45, "clutter_field_low"), "SC7b": (90, "clutter_field_high")}
SPAWNS = (
    Spawn("S1", 0.0, 0.0, 0.0, "field centre, clutter on all sides"),
    Spawn("S2", -13.0, -13.0, math.pi / 4, "south-west corner region, facing into the field"),
    Spawn("S3", 13.0, 6.0, math.pi, "east side, 2.0 m from the east wall, facing into the field"),
)

Block = tuple[int, int, int, int]  # cx, cy, sx, sy in units of 0.05 m


def _gap(a: Block, b: Block) -> float:
    dx = max(0.0, abs(a[0] - b[0]) - (a[2] + b[2]) / 2)
    dy = max(0.0, abs(a[1] - b[1]) - (a[3] + b[3]) / 2)
    return math.hypot(dx, dy)


def _spawn_distance(block: Block, sx: float, sy: float) -> float:
    """Distance in units from a spawn point to the nearest point of a block."""
    px, py = sx / UNIT, sy / UNIT
    dx = max(0.0, abs(px - block[0]) - block[2] / 2)
    dy = max(0.0, abs(py - block[1]) - block[3] / 2)
    return math.hypot(dx, dy)


def _place(seed: int) -> tuple[list[Block], int]:
    rng = np.random.Generator(np.random.PCG64(seed))
    blocks: list[Block] = []
    attempts = 0
    while len(blocks) < BLOCKS and attempts < ATTEMPTS:
        attempts += 1
        u = rng.random(4)
        sx = 10 + 2 * math.floor(16 * u[0])
        sy = 10 + 2 * math.floor(16 * u[1])
        cx = math.floor(600 * u[2]) - HALF
        cy = math.floor(600 * u[3]) - HALF
        candidate = (cx, cy, sx, sy)
        if 2 * abs(cx) + sx > 2 * (HALF - WALL_GAP) or 2 * abs(cy) + sy > 2 * (HALF - WALL_GAP):
            continue
        if any(_spawn_distance(candidate, s.x, s.y) < SPAWN_EXCLUSION for s in SPAWNS):
            continue
        if any(_gap(candidate, b) < BLOCK_GAP for b in blocks):
            continue
        blocks.append(candidate)
    return blocks, attempts


@functools.cache
def layout() -> tuple[int, int, int, tuple[Block, ...]]:
    """Return (seed, rejected seeds, attempts, blocks) for the first seed that places 90 blocks."""
    for rejected, seed in enumerate(SEEDS):
        blocks, attempts = _place(seed)
        if len(blocks) == BLOCKS:
            return seed, rejected, attempts, tuple(blocks)
    raise SystemExit(f"SC7: no seed in {SEEDS.start}..{SEEDS.stop - 1} placed {BLOCKS} blocks")


def _primitives(blocks: tuple[Block, ...]) -> list[Primitive]:
    prims = []
    for k, (cx, cy, sx, sy) in enumerate(blocks, start=1):
        x0, x1 = (cx - sx / 2) * UNIT, (cx + sx / 2) * UNIT
        y0, y1 = (cy - sy / 2) * UNIT, (cy + sy / 2) * UNIT
        prims.append(box(f"block_{k:03d}", x0, x1, y0, y1, HEIGHT, Role.OBSTACLE, "obstacle"))
    return prims


def _checks(count: int) -> ExtraCheck:
    def run(world: World, a: Analysis) -> list[CheckResult]:
        _, _, _, blocks = layout()
        mine = blocks[:count]
        results = [no_symmetry("SC7 block layout has no symmetry", world)]
        full = _primitives(blocks)
        own = [p for p in world.primitives if p.name.startswith("block_")]
        results.append(
            CheckResult(
                "SC7 blocks are the first blocks of SC7b's list",
                own == full[:count],
                len(own),
            )
        )
        gaps = [_gap(p, q) for i, p in enumerate(mine) for q in mine[i + 1 :]]
        walls = [min(HALF - abs(b[0]) - b[2] / 2, HALF - abs(b[1]) - b[3] / 2) for b in mine]
        spawns = [_spawn_distance(b, s.x, s.y) for b in mine for s in SPAWNS]
        results.append(
            CheckResult(
                "SC7 gaps ≥ 1.0 m between blocks and to the wall; ≥ 1.5 m to spawns",
                min(gaps) >= BLOCK_GAP
                and min(walls) >= WALL_GAP
                and min(spawns) >= SPAWN_EXCLUSION,
                {
                    "block_m": round(min(gaps) * UNIT, 3),
                    "wall_m": round(min(walls) * UNIT, 3),
                    "spawn_m": round(min(spawns) * UNIT, 3),
                },
            )
        )
        connected = {}
        for robot, radius in (("free", 0.0), ("mogi_0.29", 0.29), ("lynx_0.42", 0.42)):
            space = a.reachable & (a.clearance > radius) if radius else a.reachable
            connected[robot] = int(ndimage.label(space)[1])
        results.append(
            CheckResult(
                "SC7 free space is one component, also inflated by 0.29 m and 0.42 m",
                all(v == 1 for v in connected.values()),
                connected,
            )
        )
        exact = round((2 * HALF * UNIT) ** 2 - sum(b[2] * b[3] for b in mine) * UNIT * UNIT, 4)
        results.append(
            CheckResult(
                "SC7 free area equals 900.0 − Σ block areas",
                abs(a.free_area_m2 - exact) < 1e-6,
                {"raster": round(a.free_area_m2, 4), "exact": exact},
            )
        )
        return results

    return run


def build(variant: str) -> World:
    count, slug = VARIANTS[variant]
    seed, rejected, attempts, blocks = layout()
    mine = blocks[:count]
    walls, _ = walled_rectangle(-15.0, 15.0, -15.0, 15.0)
    area = round((2 * HALF * UNIT) ** 2 - sum(b[2] * b[3] for b in mine) * UNIT * UNIT, 2)
    sized = sum(1 for b in mine if max(b[2], b[3]) * UNIT >= 2.0 - 1e-9)
    return World(
        id=variant,
        slug=slug,
        title=f"Clutter field, {count} blocks",
        tier="SC",
        role="evaluation",
        primitives=(*walls, *_primitives(mine)),
        spawns=SPAWNS,
        expected=Expected(
            free_area_m2=area, narrowest_passage_m=1.0, loops=sized, loops_unsized=count
        ),
        parameters={
            "blocks": count,
            "block_side_m": [0.5, 2.0, 0.1],
            "block_height_m": HEIGHT,
            "min_gap_blocks_m": BLOCK_GAP * UNIT,
            "min_gap_wall_m": WALL_GAP * UNIT,
            "spawn_exclusion_m": SPAWN_EXCLUSION * UNIT,
            "attempt_limit": ATTEMPTS,
            "generator": "numpy.random.Generator(PCG64(seed)).random(4) per attempt",
        },
        seed=seed,
        dimensions=(Dimension((-15.0, -15.8), (15.0, -15.8), "30.0 m"),),
        sources=(
            {"what": "obstacle-field difficulty measures", "cite": "perille2020barn"},
            {"what": "cluttered scene", "cite": "xu2022explorebench"},
        ),
        extra_checks=(_checks(count),),
        plan_data={
            "seed_search": {"first": SEEDS.start, "rejected_seeds": rejected, "attempts": attempts},
            "blocks_m": [
                [round(cx * UNIT, 2), round(cy * UNIT, 2), round(sx * UNIT, 2), round(sy * UNIT, 2)]
                for cx, cy, sx, sy in mine
            ],
        },
    )


def build_a() -> World:
    return build("SC7a")


def build_b() -> World:
    return build("SC7b")
