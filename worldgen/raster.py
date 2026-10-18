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

"""0.05 m rasters, free space, clearance and the pruned skeleton (T0 §7).

Free space is 4-connected (scipy's default, as the coverage reference builder's labelling); obstacles are
8-connected, the complementary choice, so holes and loops agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
from matplotlib.path import Path as MplPath
from scipy import ndimage

from .world import BoolArray, FloatArray, Kind, Point, Primitive, polygon_area

RESOLUTION = 0.05
MARGIN_M = 1.0
SLAB = (0.03, 0.35)  # lidar slab of exploration_benchmark_runner/config/coverage/scenarios.yaml
TIP_RISE_M = 0.10
TIP_PLATEAU_M = 0.50
EIGHT = np.ones((3, 3), dtype=bool)
OFFSETS = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class Grid:
    """Cell (r, c) covers x ∈ [x0 + c·h, x0 + (c + 1)·h], y ∈ [y0 + r·h, y0 + (r + 1)·h]."""

    x0: float
    y0: float
    nx: int
    ny: int
    h: float = RESOLUTION

    @classmethod
    def around(cls, bounds: tuple[float, float, float, float], margin: float = MARGIN_M) -> Grid:
        xmin, xmax, ymin, ymax = bounds
        h = RESOLUTION
        x0 = math.floor(round((xmin - margin) / h, 6)) * h
        y0 = math.floor(round((ymin - margin) / h, 6)) * h
        nx = math.ceil(round((xmax + margin - x0) / h, 6))
        ny = math.ceil(round((ymax + margin - y0) / h, 6))
        return cls(round(x0, 6), round(y0, 6), nx, ny)

    def cell(self, x: float, y: float) -> Cell:
        return int(math.floor((y - self.y0) / self.h)), int(math.floor((x - self.x0) / self.h))

    def centre(self, r: int, c: int) -> Point:
        return self.x0 + (c + 0.5) * self.h, self.y0 + (r + 0.5) * self.h

    def centres(self) -> tuple[FloatArray, FloatArray]:
        xs: FloatArray = self.x0 + (np.arange(self.nx, dtype=np.float64) + 0.5) * self.h
        ys: FloatArray = self.y0 + (np.arange(self.ny, dtype=np.float64) + 0.5) * self.h
        return xs, ys


def rasterise(
    grid: Grid, primitives: tuple[Primitive, ...], slab: tuple[float, float] | None = SLAB
) -> BoolArray:
    """Cells whose centre lies inside a primitive footprint. With a slab, only primitives intersecting it count.

    Each primitive is tested only within its own bounding box, so large worlds stay cheap.
    """
    occ = np.zeros((grid.ny, grid.nx), dtype=bool)
    xs, ys = grid.centres()
    for p in primitives:
        if slab is not None and (p.z_min > slab[1] or p.z_max < slab[0]):
            continue
        bx0, bx1, by0, by1 = p.footprint_bounds()
        c0 = max(0, int(math.floor((bx0 - grid.x0) / grid.h)) - 1)
        c1 = min(grid.nx, int(math.ceil((bx1 - grid.x0) / grid.h)) + 1)
        r0 = max(0, int(math.floor((by0 - grid.y0) / grid.h)) - 1)
        r1 = min(grid.ny, int(math.ceil((by1 - grid.y0) / grid.h)) + 1)
        if c0 >= c1 or r0 >= r1:
            continue
        dx = xs[c0:c1][None, :] - p.x
        dy = ys[r0:r1][:, None] - p.y
        if p.kind is Kind.POLYGON:
            gx, gy = np.meshgrid(xs[c0:c1], ys[r0:r1])
            pts = np.column_stack([gx.ravel(), gy.ravel()])
            inside = MplPath(np.asarray(p.vertices, dtype=float)).contains_points(pts)
            inside = inside.reshape(gy.shape)
        elif p.kind is Kind.CYLINDER:
            inside = dx * dx + dy * dy <= p.size[0] ** 2 + 1e-12
        else:
            c, s = math.cos(p.yaw), math.sin(p.yaw)
            u = dx * c + dy * s
            v = -dx * s + dy * c
            inside = (np.abs(u) <= p.size[0] / 2 + 1e-9) & (np.abs(v) <= p.size[1] / 2 + 1e-9)
        occ[r0:r1, c0:c1] |= inside
    return occ


def polygon_mask(grid: Grid, polygons: tuple[tuple[Point, ...], ...]) -> BoolArray:
    """Cells whose centre lies inside any of the polygons (matplotlib's point-in-polygon)."""
    xs, ys = grid.centres()
    gx, gy = np.meshgrid(xs, ys)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    mask = np.zeros(grid.ny * grid.nx, dtype=bool)
    for poly in polygons:
        mask |= MplPath(np.asarray(poly, dtype=float)).contains_points(pts)
    return mask.reshape(grid.ny, grid.nx)


def component_at(mask: BoolArray, cell: Cell) -> BoolArray:
    """The 4-connected component of `mask` containing `cell` (empty if the cell is not in the mask)."""
    labels, _ = ndimage.label(mask)
    lab = labels[cell]
    if lab == 0:
        return np.zeros_like(mask)
    result: BoolArray = labels == lab
    return result


def trace_outline(grid: Grid, mask: BoolArray) -> tuple[Point, ...]:
    """The outer boundary of a mask as a rectilinear polygon along cell edges (counter-clockwise, collinear
    vertices merged). The loop with the largest area is returned."""
    ny, nx = mask.shape
    padded = np.pad(mask, 1)
    nxt: dict[tuple[int, int], list[tuple[int, int]]] = {}

    def add(a: tuple[int, int], b: tuple[int, int]) -> None:
        nxt.setdefault(a, []).append(b)

    rows, cols = np.nonzero(mask)
    for r, c in zip(rows.tolist(), cols.tolist(), strict=True):
        pr, pc = r + 1, c + 1
        if not padded[pr - 1, pc]:
            add((c, r), (c + 1, r))
        if not padded[pr, pc + 1]:
            add((c + 1, r), (c + 1, r + 1))
        if not padded[pr + 1, pc]:
            add((c + 1, r + 1), (c, r + 1))
        if not padded[pr, pc - 1]:
            add((c, r + 1), (c, r))
    loops: list[list[tuple[int, int]]] = []
    while nxt:
        start = min(nxt)
        loop = [start]
        prev_dir: tuple[int, int] | None = None
        cur = start
        while True:
            outs = nxt[cur]
            if len(outs) == 1 or prev_dir is None:
                choice = outs[0]
            else:
                # at a pinch vertex prefer the left turn, which keeps loops apart
                dx, dy = prev_dir
                here = cur
                choice = max(outs, key=lambda b: dx * (b[1] - here[1]) - dy * (b[0] - here[0]))
            outs.remove(choice)
            if not outs:
                del nxt[cur]
            prev_dir = (choice[0] - cur[0], choice[1] - cur[1])
            cur = choice
            if cur == start:
                break
            loop.append(cur)
        loops.append(loop)

    best = max(loops, key=lambda lp: abs(polygon_area(lp)))
    simplified: list[tuple[int, int]] = []
    n = len(best)
    for i in range(n):
        a, b, c = best[i - 1], best[i], best[(i + 1) % n]
        if (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]) != 0:
            simplified.append(b)
    return tuple(
        (round(grid.x0 + vx * grid.h, 4), round(grid.y0 + vy * grid.h, 4)) for vx, vy in simplified
    )


def clearance(free: BoolArray, h: float = RESOLUTION) -> FloatArray:
    """Distance from each free cell centre to the nearest non-free cell centre, in metres."""
    result: FloatArray = ndimage.distance_transform_edt(free) * h
    return result


# ---------------------------------------------------------------------------------------------------------
# Skeleton (T0 §7): Zhang–Suen thinning, a minimal skeleton, tip erosion.


def zhang_suen(mask: BoolArray) -> BoolArray:
    """Zhang–Suen thinning, vectorised: both sub-iterations repeat until nothing changes."""
    img = mask.astype(np.uint8)
    while True:
        changed = False
        for step in (0, 1):
            p = np.pad(img, 1)
            p2, p3, p4, p5 = p[:-2, 1:-1], p[:-2, 2:], p[1:-1, 2:], p[2:, 2:]
            p6, p7, p8, p9 = p[2:, 1:-1], p[2:, :-2], p[1:-1, :-2], p[:-2, :-2]
            seq = (p2, p3, p4, p5, p6, p7, p8, p9, p2)
            b = sum(s.astype(np.int16) for s in seq[:8])
            a = sum(((seq[i] == 0) & (seq[i + 1] == 1)).astype(np.int16) for i in range(8))
            if step == 0:
                cond = ((p2 * p4 * p6) == 0) & ((p4 * p6 * p8) == 0)
            else:
                cond = ((p2 * p4 * p8) == 0) & ((p2 * p6 * p8) == 0)
            remove = (img == 1) & (b >= 2) & (b <= 6) & (a == 1) & cond
            if remove.any():
                img[remove] = 0
                changed = True
        if not changed:
            thin: BoolArray = img.astype(bool)
            return thin


def _neighbours(sk: BoolArray, cell: Cell) -> list[Cell]:
    r, c = cell
    return [(r + dr, c + dc) for dr, dc in OFFSETS if sk[r + dr, c + dc]]


def _groups(cells: list[Cell]) -> int:
    """Number of 8-connected groups among a set of cells."""
    remaining = set(cells)
    groups = 0
    while remaining:
        groups += 1
        stack = [remaining.pop()]
        while stack:
            r, c = stack.pop()
            for dr, dc in OFFSETS:
                q = (r + dr, c + dc)
                if q in remaining:
                    remaining.remove(q)
                    stack.append(q)
    return groups


def _minimal(sk: BoolArray) -> BoolArray:
    """Remove, in row-major order until a pass removes nothing, cells whose (two or more) skeleton neighbours
    form one 8-connected group. `sk` is padded by one cell and is modified in place."""
    while True:
        removed = False
        rows, cols = np.nonzero(sk)
        for cell in zip(rows.tolist(), cols.tolist(), strict=True):
            nb = _neighbours(sk, cell)
            if len(nb) >= 2 and _groups(nb) == 1:
                sk[cell] = False
                removed = True
        if not removed:
            return sk


def _step(a: Cell, b: Cell, h: float) -> float:
    return h * (math.sqrt(2.0) if a[0] != b[0] and a[1] != b[1] else 1.0)


def _erode_tips(sk: BoolArray, dist: FloatArray, h: float) -> BoolArray:
    """Tip erosion: cut the start of an end branch when the clearance rises above D(e) + 0.10 m, or a junction is
    reached, within 0.50 m of the endpoint e. Junction cells are never deleted. Inputs are padded."""
    while True:
        cut = False
        rows, cols = np.nonzero(sk)
        ends = [
            q
            for q in zip(rows.tolist(), cols.tolist(), strict=True)
            if len(_neighbours(sk, q)) == 1
        ]
        for e in ends:
            if not sk[e] or len(_neighbours(sk, e)) != 1:
                continue
            path = [e]
            prev: Cell | None = None
            cur = e
            walked = 0.0
            rose = False
            while True:
                options = [n for n in _neighbours(sk, cur) if n != prev]
                if len(options) != 1:
                    break
                nxt = options[0]
                if dist[nxt] > dist[e] + TIP_RISE_M:
                    rose = True
                    break
                walked += _step(cur, nxt, h)
                prev, cur = cur, nxt
                if len(_neighbours(sk, cur)) != 2:
                    break
                path.append(cur)
                if walked >= TIP_PLATEAU_M:
                    break
            if walked < TIP_PLATEAU_M and (rose or len(_neighbours(sk, cur)) >= 3):
                for q in path:
                    if len(_neighbours(sk, q)) <= 2:
                        sk[q] = False
                cut = True
        sk = _minimal(sk)
        if not cut:
            return sk


def skeleton(free: BoolArray, dist: FloatArray, h: float = RESOLUTION) -> BoolArray:
    """`raster.skeleton`: thinning, the minimal skeleton and tip erosion of the free space (T0 §7)."""
    sk = _minimal(np.pad(zhang_suen(free), 1))
    sk = _erode_tips(sk, np.pad(dist, 1), h)
    result: BoolArray = sk[1:-1, 1:-1]
    return result


def skeleton_nodes(sk: BoolArray) -> tuple[BoolArray, BoolArray]:
    """(end cells, junction cells) of a skeleton: cells with exactly one 8-neighbour on the skeleton, and cells
    with three or more."""
    padded = np.pad(sk, 1)
    counts = (
        ndimage.convolve(padded.astype(np.int16), EIGHT.astype(np.int16), mode="constant") - padded
    )
    ends: BoolArray = (padded & (counts == 1))[1:-1, 1:-1]
    junctions: BoolArray = (padded & (counts >= 3))[1:-1, 1:-1]
    return ends, junctions


def skeleton_counts(sk: BoolArray) -> tuple[int, int]:
    """(dead ends, junctions): end cells; 8-connected clusters of junction cells (`skeleton_nodes`)."""
    ends, junctions = skeleton_nodes(sk)
    _, clusters = ndimage.label(junctions, structure=EIGHT)
    return int(np.count_nonzero(ends)), int(clusters)


def grid_graph(
    mask: BoolArray, h: float
) -> tuple[Any, npt.NDArray[np.int64], npt.NDArray[np.int64]]:
    """The 8-connected graph of a mask (edge weights h and √2·h): (sparse graph, cell index array, cells)."""
    from scipy import sparse

    index = -np.ones(mask.shape, dtype=np.int64)
    cells = np.argwhere(mask).astype(np.int64)
    index[mask] = np.arange(len(cells))
    rows, cols, weights = [], [], []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        nr, nc = cells[:, 0] + dr, cells[:, 1] + dc
        ok = (nr >= 0) & (nr < mask.shape[0]) & (nc >= 0) & (nc < mask.shape[1])
        ok[ok] = mask[nr[ok], nc[ok]]
        rows.append(index[cells[ok, 0], cells[ok, 1]])
        cols.append(index[nr[ok], nc[ok]])
        weights.append(np.full(int(ok.sum()), h * (math.sqrt(2.0) if dr and dc else 1.0)))
    n = len(cells)
    graph = sparse.csr_matrix(
        (np.concatenate(weights), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
    )
    return graph, index, cells


def geodesic(mask: BoolArray, start: Cell, goals: list[Cell], h: float = RESOLUTION) -> list[float]:
    """Shortest 8-connected path lengths (m) within the mask from `start` to each goal (inf if unreachable)."""
    from scipy.sparse import csgraph

    graph, index, _ = grid_graph(mask, h)
    dist = csgraph.dijkstra(graph, directed=False, indices=int(index[start]))
    return [float(dist[index[g]]) if index[g] >= 0 else math.inf for g in goals]
