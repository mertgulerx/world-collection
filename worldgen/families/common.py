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

"""Helpers shared by the generated families (C1–C6)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

import numpy as np

from .. import raster
from ..checks import Analysis
from ..world import BoolArray, CheckResult, Primitive, Role, World

RAY_RANGE_M = 10.0  # MOGI's lidar range
AXIS_RAYS = (0, 4, 8, 12)  # east, north, west, south
CANDIDATES = 1000  # candidates per setting, seeds SEED_BASE + 1000·j + i for i = 0…999
PILOT_OFFSET = 50000  # the pilot sequence starts this far above the evaluation sequence


class Infeasible(Exception):
    """A candidate the generator cannot complete; the message is the rejection reason."""


class SelectionFailed(RuntimeError):
    """Every candidate of a setting was rejected."""


def seed_rules(seed_base: int) -> dict[str, str]:
    """The `FamilyInfo.seeds` text of a family whose setting j tries seeds `seed_base` + 1000·j + i."""
    return {
        "evaluation": f"{seed_base} + 1000·j + i (j: setting, i: candidate 0…{CANDIDATES - 1})",
        "pilot": f"{seed_base + PILOT_OFFSET} + 1000·j + i",
    }


def streams(seed: int) -> tuple[np.random.Generator, np.random.Generator, np.random.Generator]:
    """The three independent streams of a candidate seed, in spawn order: geometry, landmarks, spawn."""
    geom, land, spawn = (
        np.random.Generator(np.random.PCG64(s)) for s in np.random.SeedSequence(seed).spawn(3)
    )
    return geom, land, spawn


def select[C](
    name: str,
    first_seed: int,
    make: Callable[[int, int, list[dict[str, object]]], C],
    reject: Callable[[C], str | None],
) -> C:
    """The first accepted candidate: seeds `first_seed` + i for i = 0…CANDIDATES − 1.

    `make(seed, i, rejected)` builds candidate i (it may raise `Infeasible`); `reject` returns the reason a
    built candidate fails, or None. Each rejection is recorded in `rejected` as {seed, reason}.
    """
    rejected: list[dict[str, object]] = []
    for i in range(CANDIDATES):
        seed = first_seed + i
        try:
            candidate = make(seed, i, rejected)
        except Infeasible as exc:
            rejected.append({"seed": seed, "reason": str(exc)})
            continue
        reason = reject(candidate)
        if reason is None:
            return candidate
        rejected.append({"seed": seed, "reason": reason})
    raise SelectionFailed(f"{name}: all {CANDIDATES} candidates rejected")


def base_reject(world: World, a: Analysis) -> str | None:
    """The acceptance rules every family shares (T0 checks 2, 3, 4 and 7 against `world.expected`).

    Free space is one closed component; the narrowest passage reads at least `minimum_passage_m` less one
    raster cell; the spawn is reachable with at least 0.5 m clearance; and the raster loop counts equal each
    expected count that is not None.
    """
    exp = world.expected
    if a.border_touch or a.sealed.any():
        return "free space not one closed component"
    if a.narrowest_m is None or a.narrowest_m < exp.minimum_passage_m - a.grid.h - 1e-9:
        return f"narrowest passage {a.narrowest_m} m below {exp.minimum_passage_m} m"
    s = world.spawns[0]
    cell = a.grid.cell(s.x, s.y)
    if not a.reachable[cell] or a.clearance[cell] < 0.5:
        return "spawn clearance below 0.5 m"
    if exp.loops is not None and a.loops != exp.loops:
        return f"raster loops {a.loops} differ from {exp.loops}"
    if exp.loops_unsized is not None and a.loops_unsized != exp.loops_unsized:
        return f"raster loops_unsized {a.loops_unsized} differ from {exp.loops_unsized}"
    return None


def first_failure(results: Iterable[CheckResult]) -> str | None:
    """The name and value of the first failed check, or None when all pass."""
    for r in results:
        if not r.passed:
            return f"{r.name}: {r.value}"
    return None


def spaced_centres(
    rng: np.random.Generator, count: int, low_m: float, high_m: float, gap: int, budget: int
) -> tuple[list[int], int]:
    """`count` centres drawn uniform in [low_m, high_m] and snapped to integer 0.1 m units, each at least `gap`
    units from those already kept; at most `budget` draws. Returns the centres in draw order and the draws
    used; raises `Infeasible("placement_failed")` when the budget runs out."""
    centres: list[int] = []
    draws = 0
    while len(centres) < count:
        draws += 1
        if draws > budget:
            raise Infeasible("placement_failed")
        c = snap_units(float(rng.uniform(low_m, high_m)))
        if all(abs(c - other) >= gap for other in centres):
            centres.append(c)
    return centres, draws


def snap(value: float) -> float:
    """Nearest 0.1 m, halves up."""
    return math.floor(value * 10 + 0.5) / 10


def snap_units(value: float) -> int:
    """Nearest 0.1 m, halves up, as an integer count of 0.1 m."""
    return math.floor(value * 10 + 0.5)


def full_height_raster(prims: Sequence[Primitive]) -> tuple[raster.Grid, BoolArray]:
    """The 0.05 m grid around the primitives and their full-height occupancy (no lidar slab)."""
    xs = [p.footprint_bounds() for p in prims]
    bounds = (
        min(b[0] for b in xs),
        max(b[1] for b in xs),
        min(b[2] for b in xs),
        max(b[3] for b in xs),
    )
    grid = raster.Grid.around(bounds)
    return grid, raster.rasterise(grid, tuple(prims), slab=None)


def longest_ray_heading(
    grid: raster.Grid, occ: BoolArray, x: float, y: float, rays: Sequence[int]
) -> float:
    """Yaw k·π/8 of the longest free ray among `rays`, cast up to 10 m on the full-height raster; ties go to the
    first ray in `rays`."""
    step = grid.h / 4
    best_k, best_len = rays[0], -1.0
    for k in rays:
        yaw = k * math.pi / 8
        length = 0.0
        t = step
        while t <= RAY_RANGE_M + 1e-9:
            r, c = grid.cell(x + t * math.cos(yaw), y + t * math.sin(yaw))
            if not (0 <= r < grid.ny and 0 <= c < grid.nx) or occ[r, c]:
                break
            length = t
            t += step
        length = round(length, 3)
        if length > best_len:
            best_k, best_len = k, length
    return best_k * math.pi / 8


def _overlap(p: Primitive, q: Primitive) -> float:
    a, b = p.footprint_bounds(), q.footprint_bounds()
    dx = min(a[1], b[1]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[2], b[2])
    return dx * dy if dx > 1e-9 and dy > 1e-9 else 0.0


def _on_grid(v: float) -> bool:
    return abs(v * 10 - round(v * 10)) < 1e-6


def walls_check(
    family: str, world: World, roles: tuple[Role, ...] = (Role.WALL, Role.BOUNDARY_WALL)
) -> CheckResult:
    """Boxes of the given roles (walls by default) are pairwise disjoint and lie on the 0.1 m grid."""
    walls = [p for p in world.primitives if p.role in roles]
    overlap = sum(_overlap(p, q) for i, p in enumerate(walls) for q in walls[i + 1 :])
    gridded = all(_on_grid(v) for p in walls for v in p.footprint_bounds())
    return CheckResult(
        f"{family} wall boxes disjoint and on the 0.1 m grid",
        overlap == 0.0 and gridded,
        {"boxes": len(walls), "overlap_m2": round(overlap, 6), "on_grid": gridded},
    )


def seeded_spawn(
    prims: Sequence[Primitive],
    inside: tuple[float, float],
    rng: np.random.Generator,
    rays: Sequence[int] = AXIS_RAYS,
) -> tuple[float, float, float]:
    """A spawn (x, y, yaw) on the full-height raster: the free cells connected to `inside` with clearance of at
    least 0.5 m, in row-major order, index `rng.integers(n)`; heading by `longest_ray_heading`."""
    grid, occ = full_height_raster(prims)
    reachable = raster.component_at(~occ, grid.cell(*inside))
    dist = raster.clearance(reachable, grid.h)
    cells = np.argwhere(reachable & (dist >= 0.5))
    r, c = (int(v) for v in cells[int(rng.integers(len(cells)))])
    x, y = (round(v, 6) for v in grid.centre(r, c))
    return x, y, longest_ray_heading(grid, occ, x, y, rays)
