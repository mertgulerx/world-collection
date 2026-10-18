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

"""Automatic checks of a built world (T0 §7). Each check passes or fails with a value."""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import ndimage

from . import raster
from .raster import Cell, Grid
from .world import MIN_HEIGHT, MOGI_RADIUS_M, BoolArray, CheckResult, FloatArray, World

GZ_TIMEOUT_S = 60.0


@dataclass(frozen=True, slots=True)
class Analysis:
    """Everything the checks, the plan image and the descriptors need, computed once."""

    grid: Grid
    occupied: BoolArray  # lidar-slab footprint
    inclusion: BoolArray
    inclusion_polygons: tuple[tuple[tuple[float, float], ...], ...]
    reachable: BoolArray
    sealed: BoolArray  # free cells inside the inclusion that the spawns cannot reach
    clearance: FloatArray
    skeleton: BoolArray
    narrowest_m: float | None
    narrowest_cell: Cell | None
    loops: int
    loops_unsized: int
    border_touch: bool

    @property
    def free_area_m2(self) -> float:
        return float(np.count_nonzero(self.reachable)) * self.grid.h**2


def analyse(world: World) -> Analysis:
    grid = raster.Grid.around(world.bounds())
    statics = world.static_primitives()
    occupied = raster.rasterise(grid, statics)
    spawn = world.spawns[0]
    start = grid.cell(spawn.x, spawn.y)
    open_space = ~occupied
    polygons: tuple[tuple[tuple[float, float], ...], ...]
    if world.inclusion is None:
        around = raster.component_at(open_space, start)
        inclusion = ndimage.binary_fill_holes(around)
        polygons = (raster.trace_outline(grid, inclusion),) if inclusion.any() else ()
    else:
        polygons = world.inclusion
        inclusion = raster.polygon_mask(grid, polygons)
    free = inclusion & open_space
    reachable = raster.component_at(free, start)
    sealed = free & ~reachable
    border = bool(
        reachable[0, :].any()
        or reachable[-1, :].any()
        or reachable[:, 0].any()
        or reachable[:, -1].any()
    )
    dist = raster.clearance(reachable, grid.h)
    sk = raster.skeleton(reachable, dist, grid.h) if reachable.any() else np.zeros_like(reachable)
    if sk.any():
        masked = np.where(sk, dist, np.inf)
        idx = np.unravel_index(int(np.argmin(masked)), masked.shape)
        cell = (int(idx[0]), int(idx[1]))
        narrowest = round(2.0 * float(dist[cell]), 4)
    else:
        cell, narrowest = None, None
    loops, unsized = count_loops(occupied, inclusion, grid.h)
    return Analysis(
        grid,
        occupied,
        inclusion,
        polygons,
        reachable,
        sealed,
        dist,
        sk,
        narrowest,
        cell,
        loops,
        unsized,
        border,
    )


def count_loops(occupied: BoolArray, inclusion: BoolArray, h: float) -> tuple[int, int]:
    """T0 check 7: (`loops`, `loops_unsized`).

    Obstacles are 8-connected components of un-inflated occupied cells; their size is the longer side of the
    axis-aligned bounding box. Obstacles are inflated by MOGI's circumscribed radius (`MOGI_RADIUS_M`). An
    inflated component is enclosed when all its cells lie inside the inclusion region (the boundary walls do
    not).
    """
    inflated = ndimage.distance_transform_edt(~occupied) * h <= MOGI_RADIUS_M
    labels, n = ndimage.label(inflated, structure=raster.EIGHT)
    obstacles, n_obs = ndimage.label(occupied, structure=raster.EIGHT)
    slices = ndimage.find_objects(obstacles)
    big = np.zeros(n_obs + 1, dtype=bool)
    for i, sl in enumerate(slices, start=1):
        if sl is not None:
            side = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * h
            big[i] = side >= 2.0 - 1e-9
    outside = np.unique(labels[inflated & ~inclusion])
    sized = unsized = 0
    for lab in range(1, n + 1):
        if lab in outside:
            continue
        unsized += 1
        members = np.unique(obstacles[(labels == lab) & occupied])
        if big[members[members > 0]].any():
            sized += 1
    return sized, unsized


def gz_check(sdf_path: Path) -> CheckResult:
    """Check 1: `gz sdf -k` with the ROS environment sourced (a parser, not the simulator)."""
    cmd = [
        "bash",
        "-c",
        'source /opt/ros/jazzy/setup.bash && gz sdf -k "$1"',
        "bash",
        str(sdf_path),
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=GZ_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return CheckResult("1 gz sdf -k", False, None, f"could not run: {exc}")
    text = (out.stdout + out.stderr).strip()
    ok = out.returncode == 0 and "Valid" in text
    first = text.splitlines()[0] if text else ""
    return CheckResult(
        "1 gz sdf -k", ok, first if ok else f"exit {out.returncode}", "" if ok else first[:300]
    )


def run_checks(world: World, a: Analysis, sdf_path: Path) -> list[CheckResult]:
    exp = world.expected
    h = a.grid.h
    results = [gz_check(sdf_path)]
    sealed_m2 = float(np.count_nonzero(a.sealed)) * h * h
    results.append(
        CheckResult(
            "2 boundary closed, no sealed pockets",
            not a.border_touch and sealed_m2 == 0.0,
            {"touches_border": a.border_touch, "sealed_m2": round(sealed_m2, 4)},
        )
    )
    spawn_values = {}
    spawn_ok = True
    for s in world.spawns:
        cell = a.grid.cell(s.x, s.y)
        inside = bool(a.reachable[cell])
        clear = round(float(a.clearance[cell]), 3) if inside else 0.0
        spawn_values[s.id] = clear
        spawn_ok &= inside and clear >= 0.5
    results.append(CheckResult("3 spawns reachable with 0.5 m clearance", spawn_ok, spawn_values))
    narrow_ok = a.narrowest_m is not None and a.narrowest_m >= exp.minimum_passage_m - h - 1e-9
    narrow_tol = h if exp.narrowest_tolerance_m is None else exp.narrowest_tolerance_m
    match = (
        a.narrowest_m is not None
        and abs(a.narrowest_m - exp.narrowest_passage_m) <= narrow_tol + 1e-9
    )
    where = a.grid.centre(*a.narrowest_cell) if a.narrowest_cell else None
    results.append(
        CheckResult(
            "4 narrowest passage",
            narrow_ok and match,
            a.narrowest_m,
            f"expected {exp.narrowest_passage_m} ± {narrow_tol:g} m (minimum {exp.minimum_passage_m} m) at "
            f"{tuple(round(v, 2) for v in where) if where else None}",
        )
    )
    low = [
        f"{p.name} {p.height} m < {MIN_HEIGHT[p.role]} m"
        for p in world.primitives
        if MIN_HEIGHT[p.role] is not None and p.height < float(MIN_HEIGHT[p.role] or 0.0) - 1e-9
    ]
    results.append(CheckResult("5 heights by role", not low, len(low), "; ".join(low[:5])))
    area = round(a.free_area_m2, 2)
    tol = exp.free_area_tolerance
    results.append(
        CheckResult(
            "6 free area",
            abs(area - exp.free_area_m2) <= tol * exp.free_area_m2,
            area,
            f"expected {exp.free_area_m2} m² ± {tol:.0%}",
        )
    )
    loop_ok = True
    detail = []
    if exp.loops is not None:
        loop_ok &= a.loops == exp.loops
        detail.append(f"loops expected {exp.loops}")
    if exp.loops_unsized is not None:
        loop_ok &= a.loops_unsized == exp.loops_unsized
        detail.append(f"loops_unsized expected {exp.loops_unsized}")
    results.append(
        CheckResult(
            "7 loops",
            loop_ok,
            {"loops": a.loops, "loops_unsized": a.loops_unsized},
            "; ".join(detail),
        )
    )
    for extra in world.extra_checks:
        results.extend(extra(world, a))
    return results


def no_symmetry(name: str, world: World) -> CheckResult:
    """A world-specific helper: the structure differs from its mirror images and its 180° (and 90°) rotations
    about the world origin. It is rasterised on a grid centred on the origin, so equal arrays mean symmetry."""
    xmin, xmax, ymin, ymax = world.bounds()
    half = max(abs(xmin), abs(xmax), abs(ymin), abs(ymax)) + raster.MARGIN_M
    cells = int(np.ceil(round(half / raster.RESOLUTION, 6)))
    grid = Grid(-cells * raster.RESOLUTION, -cells * raster.RESOLUTION, 2 * cells, 2 * cells)
    mask = raster.rasterise(grid, world.static_primitives())
    variants = {
        "mirror x": mask[:, ::-1],
        "mirror y": mask[::-1, :],
        "rotation 180°": mask[::-1, ::-1],
        "rotation 90°": np.rot90(mask),
        "mirror diagonal": mask.T,
    }
    same = [k for k, v in variants.items() if np.array_equal(v, mask)]
    return CheckResult(name, not same, same or "none", "symmetries found" if same else "")
