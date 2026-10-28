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

"""World descriptors `world-descriptors-v1`.

All values come from the reachable free space F of the 0.05 m raster and its clearance D. Sampled values use one
numpy PCG64 generator per world, `SeedSequence([2026])`, drawing in a fixed order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage
from scipy.sparse import csgraph

from . import raster
from .checks import Analysis
from .rng import generator
from .world import MOGI_RADIUS_M, BoolArray, FloatArray, World

VERSION = "world-descriptors-v1"
SEED = 2026
N_POINTS = 400
N_SOURCES, N_TARGETS = 40, 10
PAIR_MIN_M = 2.0
PAIR_DRAWS = 200
DISPERSION_R_M = 2.5  # obstacle range of the shared Nav2 costmaps
CORE_DEPTH_M = 0.8  # regions are separated by openings narrower than 1.6 m
REGION_MIN_M2 = 4.0
LARGE_GRID_CELLS = 2_000_000


@dataclass(frozen=True, slots=True)
class Robot:
    id: str
    radius_m: float
    range_m: float


ROBOTS = (Robot("mogi", MOGI_RADIUS_M, 10.0),)


def _sample_cells(
    rng: np.random.Generator, mask: BoolArray, n: int
) -> np.ndarray[Any, np.dtype[np.int64]]:
    cells = np.argwhere(mask)
    if len(cells) == 0:
        return np.zeros((0, 2), dtype=np.int64)
    picks = rng.choice(len(cells), size=min(n, len(cells)), replace=len(cells) < n)
    result: np.ndarray[Any, np.dtype[np.int64]] = cells[picks].astype(np.int64)
    return result


def _cells(array: np.ndarray[Any, np.dtype[np.int64]]) -> list[tuple[int, int]]:
    return [(int(array[i, 0]), int(array[i, 1])) for i in range(array.shape[0])]


def _ray_lengths(
    free: BoolArray, cell: tuple[int, int], angles: FloatArray, r_max: float, h: float
) -> FloatArray:
    """Distance from the cell centre along each angle to the first non-free cell, capped at r_max."""
    steps = np.arange(1, int(math.ceil(r_max / (h / 2))) + 1, dtype=np.float64) * (h / 2)
    r0, c0 = cell
    rr = np.floor(r0 + 0.5 + np.outer(np.sin(angles), steps) / h).astype(np.int64)
    cc = np.floor(c0 + 0.5 + np.outer(np.cos(angles), steps) / h).astype(np.int64)
    inside = (rr >= 0) & (rr < free.shape[0]) & (cc >= 0) & (cc < free.shape[1])
    blocked = ~inside
    blocked[inside] = ~free[rr[inside], cc[inside]]
    first = np.where(blocked.any(axis=1), blocked.argmax(axis=1), len(steps))
    lengths: FloatArray = np.where(
        first < len(steps), steps[np.minimum(first, len(steps) - 1)], r_max
    )
    return lengths


def _visible_cells(free: BoolArray, cell: tuple[int, int], r_max: float, h: float) -> int:
    """Free cells within r_max reached by a ray before any blocking cell; rays h/(2 r_max) apart."""
    n_rays = int(math.ceil(2 * math.pi * r_max / (h / 2)))
    angles = np.arange(n_rays, dtype=np.float64) * (2 * math.pi / n_rays)
    steps = np.arange(0, int(math.ceil(r_max / (h / 2))) + 1, dtype=np.float64) * (h / 2)
    r0, c0 = cell
    rr = np.floor(r0 + 0.5 + np.outer(np.sin(angles), steps) / h).astype(np.int64)
    cc = np.floor(c0 + 0.5 + np.outer(np.cos(angles), steps) / h).astype(np.int64)
    inside = (rr >= 0) & (rr < free.shape[0]) & (cc >= 0) & (cc < free.shape[1])
    ok: BoolArray = np.zeros_like(inside)
    ok[inside] = free[rr[inside], cc[inside]]
    seen = np.cumprod(ok, axis=1).astype(bool)  # a ray stops at its first blocking cell
    flat = rr[seen] * free.shape[1] + cc[seen]
    return int(np.unique(flat).size)


def _median_halfwidth(values: FloatArray) -> float:
    """Half-width of the 95 % order-statistic interval of the median."""
    n = len(values)
    if n < 3:
        return float("nan")
    s = np.sort(values)
    lo = max(0, int(math.floor(n / 2 - 1.96 * math.sqrt(n) / 2)))
    hi = min(n - 1, int(math.ceil(n / 2 + 1.96 * math.sqrt(n) / 2)))
    return float((s[hi] - s[lo]) / 2)


def _nav_complexity(
    free: BoolArray, rng: np.random.Generator, h: float
) -> tuple[FloatArray, float]:
    """Geodesic (8-connected) over octile distance for sampled pairs; a 0.10 m grid for very large worlds."""
    grid_h = h
    mask = free
    if free.size > LARGE_GRID_CELLS:
        ny, nx = (free.shape[0] // 2) * 2, (free.shape[1] // 2) * 2
        blocks = free[:ny, :nx].reshape(ny // 2, 2, nx // 2, 2)
        mask = np.asarray(blocks.all(axis=(1, 3)), dtype=bool)
        grid_h = 2 * h
    graph, index, cells = raster.grid_graph(mask, grid_h)
    n = len(cells)
    sources = cells[rng.choice(n, size=min(N_SOURCES, n), replace=False)]
    ratios: list[float] = []
    for sr, sc in sources:
        # one source at a time: the same distances, without a sources × cells matrix (7 GB for LS1)
        dist = csgraph.dijkstra(graph, directed=False, indices=int(index[sr, sc]))
        got = 0
        for _ in range(PAIR_DRAWS):
            tr, tc = cells[int(rng.integers(n))]
            dy, dx = abs(int(tr) - int(sr)), abs(int(tc) - int(sc))
            if math.hypot(dx, dy) * grid_h < PAIR_MIN_M:
                continue
            octile = grid_h * (max(dx, dy) + (math.sqrt(2.0) - 1.0) * min(dx, dy))
            geodesic = float(dist[index[tr, tc]])
            if math.isfinite(geodesic):
                ratios.append(geodesic / octile)
            got += 1
            if got == N_TARGETS:
                break
    return np.asarray(ratios, dtype=np.float64), grid_h


def _min_gap(free: BoolArray, h: float) -> float | None:
    """Smallest surface distance between two different obstacle components (8-connected non-free cells)."""
    labels, n = ndimage.label(~free, structure=raster.EIGHT)
    if n < 2:
        return None
    _, (ir, ic) = ndimage.distance_transform_edt(free, return_indices=True)
    nearest = labels[ir, ic]
    best = math.inf
    for axis in (0, 1):
        a = (slice(None, -1), slice(None)) if axis == 0 else (slice(None), slice(None, -1))
        b = (slice(1, None), slice(None)) if axis == 0 else (slice(None), slice(1, None))
        pair = free[a] & free[b] & (nearest[a] != nearest[b])
        if pair.any():
            d = np.hypot(ir[a][pair] - ir[b][pair], ic[a][pair] - ic[b][pair]) * h - h
            best = min(best, float(d.min()))
    return round(best, 4) if math.isfinite(best) else None


def regions(
    free: BoolArray, dist: FloatArray, h: float
) -> tuple[int, np.ndarray[Any, np.dtype[np.int32]]]:
    """Regions separated by openings narrower than 1.6 m: cores D ≥ 0.8 m grown by watershed (T1 §4.2)."""
    cores, n = ndimage.label(free & (dist >= CORE_DEPTH_M))
    if n == 0:
        return 0, np.zeros(free.shape, dtype=np.int32)
    inp = np.where(free, np.clip(1000 - dist / h, 0, 65000), 65535).astype(np.uint16)
    markers = cores.astype(np.int32)
    markers[~free] = -1
    grown: np.ndarray[Any, np.dtype[np.int32]] = ndimage.watershed_ift(inp, markers)
    grown[~free] = 0
    areas = np.bincount(grown[free].ravel(), minlength=n + 1)[1:] * h * h
    return int(np.count_nonzero(areas >= REGION_MIN_M2)), grown


def compute(world: World, a: Analysis) -> dict[str, Any]:
    h = a.grid.h
    free = a.reachable
    dist = a.clearance
    rng = generator(SEED)
    area = float(np.count_nonzero(free)) * h * h
    clear = dist[free]
    points = _sample_cells(rng, free, N_POINTS)
    sixteen = np.arange(16, dtype=np.float64) * (math.pi / 8)
    diag = math.hypot(*free.shape) * h
    char_dims, dispersions = [], []
    for r, c in _cells(points):
        lengths = _ray_lengths(free, (int(r), int(c)), sixteen, diag, h)
        char_dims.append(min(lengths[k] + lengths[k + 8] for k in range(8)))
        blocked = _ray_lengths(free, (int(r), int(c)), sixteen, DISPERSION_R_M, h) < DISPERSION_R_M
        dispersions.append(int(np.count_nonzero(blocked != np.roll(blocked, 1))))
    char_arr = np.asarray(char_dims, dtype=np.float64)
    disp_arr = np.asarray(dispersions, dtype=np.float64)
    robot_points = {
        rb.id: _sample_cells(rng, free & (dist >= rb.radius_m), N_POINTS) for rb in ROBOTS
    }
    ratios, nav_grid = _nav_complexity(free, rng, h)
    dead_ends, junctions = raster.skeleton_counts(a.skeleton)
    n_regions, _ = regions(free, dist, h)
    robots: dict[str, Any] = {}
    for rb in ROBOTS:
        vis = np.asarray(
            [
                _visible_cells(free, (int(r), int(c)), rb.range_m, h) * h * h
                for r, c in _cells(robot_points[rb.id])
            ]
        )
        spawn_vis = {}
        for s in world.spawns:
            cell = a.grid.cell(s.x, s.y)
            spawn_vis[s.id] = round(_visible_cells(free, cell, rb.range_m, h) * h * h / area, 4)
        robots[rb.id] = {
            "radius_m": rb.radius_m,
            "r_eff_m": rb.range_m,
            "visible_area_m2": {
                "mean": round(float(vis.mean()), 2),
                "n": int(len(vis)),
                "mc_se": round(float(vis.std(ddof=1) / math.sqrt(len(vis))), 2),
            },
            "visible_fraction": round(float(vis.mean()) / area, 4),
            "area_over_pi_r2": round(area / (math.pi * rb.range_m**2), 4),
            "spawn_visible_fraction": spawn_vis,
        }
    return {
        "version": VERSION,
        "grid": {"source": "t0_raster", "resolution_m": h, "seed": SEED},
        "area_m2": round(area, 2),
        "clearance_m": {
            "median": round(float(np.median(clear)), 3),
            "p10": round(float(np.percentile(clear, 10)), 3),
            "p90": round(float(np.percentile(clear, 90)), 3),
        },
        "clearance_max_m": round(float(clear.max()), 3),
        "char_dim_m": {
            "median": round(float(np.median(char_arr)), 3),
            "n": int(len(char_arr)),
            "mc_halfwidth": round(_median_halfwidth(char_arr), 3),
        },
        "dispersion": {
            "mean": round(float(disp_arr.mean()), 3),
            "n": int(len(disp_arr)),
            "mc_se": round(float(disp_arr.std(ddof=1) / math.sqrt(len(disp_arr))), 3),
        },
        "nav_complexity": {
            "median": round(float(np.median(ratios)), 4) if len(ratios) else None,
            "p90": round(float(np.percentile(ratios, 90)), 4) if len(ratios) else None,
            "n": int(len(ratios)),
            "grid_m": nav_grid,
        },
        "narrowest_passage_m": a.narrowest_m,
        "min_gap_m": _min_gap(free, h),
        "loops": a.loops,
        "loops_unsized": a.loops_unsized,
        "dead_ends": dead_ends,
        "junctions": junctions,
        "regions": n_regions,
        "robots": robots,
    }
