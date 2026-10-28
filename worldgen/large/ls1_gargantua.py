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

"""LS1 Gargantua: a 500 × 500 m office maze in which no two rooms or corridors are alike (plan LS1).

Lengths are integers in units of 0.1 m measured from the interior's south-west corner, so every wall lies on the
0.1 m construction grid. Binary space partitioning cuts the interior into districts (0.6 m walls), and each
district into cells (0.2 m walls): corridors 6.2-10.2 m between wall lines, and rooms. Each district has its own
office-maze mix between 0 (larger rooms off corridors, mostly doors) and 1 (smaller rooms, corridor labyrinths,
more loops). A random spanning tree over the walls that cells share makes every cell reachable, joining districts
through as few openings as it needs, and extra openings add loops; every opening is at least 3.0 m wide. Rooms of
at least 12 × 12 m get wall stubs and free-standing partitions that keep 3.5 m from everything else in the room.
Wall pieces shorter than 2.5 m left free-standing by removed walls are dropped, and a cell that looks like an
earlier one on 1 m blocks (rotations and mirror images included) gets a stub of its own, so no two cells are alike.
"""

from __future__ import annotations

import functools
import hashlib
import math
import statistics
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from .. import raster
from ..checks import Analysis, no_symmetry
from ..families.common import AXIS_RAYS, Infeasible, longest_ray_heading, select, streams
from ..geometry import grid_walls, walled_rectangle
from ..world import (
    BoolArray,
    CheckResult,
    Dimension,
    Expected,
    ExtraCheck,
    Primitive,
    Role,
    Spawn,
    World,
)

Axis = Literal["v", "h"]
Kind = Literal["door", "wide", "full"]
Rect = tuple[int, int, int, int]  # x0, x1, y0, y1 in units: cells [x0, x1) × [y0, y1)
Pieces = tuple[tuple[tuple[int, int], ...], ...]  # per line, its solid stretches along it
Slices = tuple[slice, slice]  # rows, columns

WORLD_ID = "LS1"
SEED_BASE = 900000  # candidate i uses seed SEED_BASE + i
UNIT = 0.1  # metres per unit
N = 5000  # interior side (500 m)
HALF_M = N * UNIT / 2
BOUNDARY = -1  # the side of a node that lies on the outer wall
WALL_HALF = 1  # cell walls, stubs and partitions 0.2 m thick
MAIN_HALF = 3  # district walls 0.6 m thick
BOUNDARY_T = 6  # outer wall 0.6 m thick
WALL_H_M = 2.0
BOUNDARY_H_M = 3.0
DISTRICT_MAX = 1600  # districts are split until no side exceeds 160 m
DISTRICT_SPLIT = (0.32, 0.68)
DOOR = (30, 50)  # door widths, 3.0-5.0 m
WIDE = (60, 140)  # wide openings, 6.0-14.0 m
MARGIN = 5  # wall kept between a door or wide opening and the ends of its shared interval (0.5 m)
EDGE_MIN = DOOR[0] + 2 * MARGIN  # shortest shared interval that can hold an opening
CLEAR = 35  # stubs and partitions keep 3.5 m from everything else in their room
STRIP = (62, 102)  # corridor line spacing, 6.2-10.2 m
CORRIDOR_LEN = (300, 1000)  # corridor segments 30-100 m
# per-district ranges as (office, maze): the value at mix m is office + (maze - office)·m
ROOM_MIN = (100, 80)  # smallest room line spacing
TARGET_LO = (160, 100)  # a node larger than a target drawn from [lo, hi] is split again
TARGET_HI = (340, 200)
P_STRIP = (0.45, 0.75)  # chance that a split cuts a corridor off a node's long side
BRAID = (0.10, 0.22)  # extra openings per cell of the district
P_FULL_CORRIDOR = (0.55, 0.85)  # corridor to corridor: wall removed (else a wide opening)
P_DOOR_CORRIDOR = (0.80, 0.45)  # room to corridor: door (else a wide opening)
P_DOOR_ROOM = (0.65, 0.35)  # room to room: door; otherwise wide (60 %) or wall removed (40 %)
P_DOOR_DISTRICT = 0.6  # across a district wall: door (else a wide opening)
STUBS = (0.12, 0.30)  # wall stubs per 100 m² of a furnished room
STUBS_MAX = 6
PARTITIONS = (0.04, 0.10)  # free-standing partitions per 100 m²
PARTITIONS_MAX = 3
FURNISH_MIN = 120  # rooms furnished when both interior sides are at least 12 m
STUB_LEN = (20, 0.45)  # stub length from 2 m to 45 % of the room's depth
PARTITION_LEN = (30, 120)  # partitions 3-12 m, at most 60 % of the room's side
ATTEMPTS = 40
FRAGMENT_MAX = (
    25  # free-standing wall pieces shorter than 2.5 m (left by full openings) are removed
)
BLOCK = 10  # cells are told apart on 1 m blocks
SPAWN_CLEAR_M = 2.0
HEADING_WINDOW_M = 12.0
PLAN_DETAIL_M = 100.0


def _m(units: float) -> float:
    """Units to metres, rounded to 0.1 mm so that no float noise reaches `world.yaml`."""
    return round(units * UNIT, 4)


def _mix(pair: tuple[float, float], mix: float) -> float:
    return pair[0] + (pair[1] - pair[0]) * mix


@dataclass(frozen=True, slots=True)
class Line:
    """A wall line: on x = `c` running along y (`axis` "v"), or on y = `c` along x ("h"), between the lines it ends
    on at `start` and `end`, into which it extends by `ext0` and `ext1` (their half thickness). It occupies
    [c − half, c + half) across."""

    axis: Axis
    c: int
    start: int
    end: int
    half: int
    ext0: int
    ext1: int
    main: bool  # a district wall


@dataclass(frozen=True, slots=True)
class Node:
    """A rectangle between partition lines; its sides are line indices or BOUNDARY."""

    x0: int
    x1: int
    y0: int
    y1: int
    w: int
    e: int
    s: int
    n: int


@dataclass(frozen=True, slots=True)
class Cell:
    node: Node
    district: int
    corridor: bool


@dataclass(frozen=True, slots=True)
class Edge:
    """Two cells (a < b) that share line `line` over the interior interval [lo, hi)."""

    a: int
    b: int
    line: int
    lo: int
    hi: int


@dataclass(frozen=True, slots=True)
class Opening:
    """A cut [start, start + width) along an edge's line; "full" removes the whole shared interval, without
    margins, so the line remains only inside the walls at its ends."""

    edge: int
    start: int
    width: int
    kind: Kind


@dataclass(frozen=True, slots=True)
class Layout:
    districts: tuple[Node, ...]
    mixes: tuple[float, ...]
    lines: tuple[Line, ...]
    cells: tuple[Cell, ...]
    edges: tuple[Edge, ...]
    tree: tuple[int, ...]
    extra: tuple[int, ...]
    openings: tuple[Opening, ...]
    pieces: Pieces
    stubs: tuple[tuple[int, Rect], ...]  # (cell, rectangle), the distinguishing stubs last
    partitions: tuple[tuple[int, Rect], ...]
    distinguishing: int  # stubs added to cells that matched an earlier cell on 1 m blocks
    fragments: int  # free-standing wall pieces shorter than FRAGMENT_MAX removed
    main: BoolArray  # district walls on the 0.1 m grid (row: y, column: x)
    wall: BoolArray  # cell walls, stubs and partitions, disjoint from `main`
    components: int  # 4-connected components of the free space
    area_m2: float


def _half(lines: list[Line], lid: int) -> int:
    return 0 if lid == BOUNDARY else lines[lid].half


def _split(node: Node, axis: Axis, pos: int, lines: list[Line], main: bool) -> tuple[Node, Node]:
    """Split `node` by a new line at `pos` (x for axis "v", y for "h"); the low part first."""
    half = MAIN_HALF if main else WALL_HALF
    if axis == "v":
        ext0, ext1 = _half(lines, node.s), _half(lines, node.n)
        lines.append(Line("v", pos, node.y0, node.y1, half, ext0, ext1, main))
        lid = len(lines) - 1
        return (
            Node(node.x0, pos, node.y0, node.y1, node.w, lid, node.s, node.n),
            Node(pos, node.x1, node.y0, node.y1, lid, node.e, node.s, node.n),
        )
    ext0, ext1 = _half(lines, node.w), _half(lines, node.e)
    lines.append(Line("h", pos, node.x0, node.x1, half, ext0, ext1, main))
    lid = len(lines) - 1
    return (
        Node(node.x0, node.x1, node.y0, pos, node.w, node.e, node.s, lid),
        Node(node.x0, node.x1, pos, node.y1, node.w, node.e, lid, node.n),
    )


def _districts(geom: np.random.Generator, lines: list[Line]) -> list[Node]:
    """Split the interior across its longer side at 32-68 % (whole metres) until no side exceeds 160 m."""
    done: list[Node] = []
    stack = [Node(0, N, 0, N, BOUNDARY, BOUNDARY, BOUNDARY, BOUNDARY)]
    while stack:
        node = stack.pop()
        w, h = node.x1 - node.x0, node.y1 - node.y0
        if max(w, h) <= DISTRICT_MAX:
            done.append(node)
            continue
        axis: Axis = "v" if w >= h else "h"
        lo, length = (node.x0, w) if axis == "v" else (node.y0, h)
        pos = lo + 10 * round(float(geom.uniform(*DISTRICT_SPLIT)) * length / 10)
        first, second = _split(node, axis, pos, lines, main=True)
        stack += [second, first]
    return sorted(done, key=lambda d: (d.y0, d.x0))


def _cells(
    district: Node, mix: float, geom: np.random.Generator, lines: list[Line]
) -> list[tuple[Node, bool]]:
    """Rooms and corridors of one district: (node, is a corridor)."""
    room_min = round(_mix(ROOM_MIN, mix))
    t_lo, t_hi = _mix(TARGET_LO, mix), _mix(TARGET_HI, mix)
    p_strip = _mix(P_STRIP, mix)
    out: list[tuple[Node, bool]] = []
    stack: list[tuple[Node, bool]] = [(district, False)]
    while stack:
        node, corridor = stack.pop()
        w, h = node.x1 - node.x0, node.y1 - node.y0
        axis: Axis
        if corridor:
            # a corridor is cut across its length into segments of CORRIDOR_LEN
            length = max(w, h)
            axis = "v" if w > h else "h"
            target = int(geom.integers(CORRIDOR_LEN[0], CORRIDOR_LEN[1] + 1))
            if length <= target or length < 2 * CORRIDOR_LEN[0]:
                out.append((node, True))
                continue
            lo = node.x0 if axis == "v" else node.y0
            pos = lo + int(geom.integers(CORRIDOR_LEN[0], length - CORRIDOR_LEN[0] + 1))
            first, second = _split(node, axis, pos, lines, main=False)
            stack += [(second, True), (first, True)]
            continue
        if max(w, h) <= float(geom.uniform(t_lo, t_hi)):
            out.append((node, False))
            continue
        long_x = w > h or (w == h and geom.random() < 0.5)
        if geom.random() < p_strip and min(w, h) >= STRIP[1] + room_min:
            # a corridor along the long side, at one of its two edges
            spacing = int(geom.integers(STRIP[0], STRIP[1] + 1))
            axis = "h" if long_x else "v"
            lo, hi = (node.y0, node.y1) if axis == "h" else (node.x0, node.x1)
            low = geom.random() < 0.5
            first, second = _split(node, axis, lo + spacing if low else hi - spacing, lines, False)
            stack += [(second, not low), (first, low)]
            continue
        axis = "v" if long_x else "h"
        lo, length = (node.x0, w) if axis == "v" else (node.y0, h)
        a_min = max(room_min, round(0.3 * length))
        a_max = min(length - room_min, round(0.7 * length))
        if a_min > a_max:
            out.append((node, False))
            continue
        first, second = _split(node, axis, lo + int(geom.integers(a_min, a_max + 1)), lines, False)
        stack += [(second, False), (first, False)]
    return out


def _interior(node: Node, lines: list[Line]) -> Rect:
    return (
        node.x0 + _half(lines, node.w),
        node.x1 - _half(lines, node.e),
        node.y0 + _half(lines, node.s),
        node.y1 - _half(lines, node.n),
    )


def _edges(cells: list[Cell], lines: list[Line]) -> list[Edge]:
    """Every pair of cells on opposite sides of a line whose interiors share at least EDGE_MIN along it."""
    interiors = [_interior(c.node, lines) for c in cells]
    low: dict[int, list[int]] = {}  # cells whose east or north side is the line
    high: dict[int, list[int]] = {}  # cells whose west or south side is the line
    for i, c in enumerate(cells):
        nd = c.node
        for lid, side in ((nd.w, high), (nd.e, low), (nd.s, high), (nd.n, low)):
            if lid != BOUNDARY:
                side.setdefault(lid, []).append(i)
    edges: list[Edge] = []
    for lid in sorted(low):
        vertical = lines[lid].axis == "v"
        for i in low[lid]:
            for j in high.get(lid, []):
                p, q = interiors[i], interiors[j]
                lo, hi = (
                    (max(p[2], q[2]), min(p[3], q[3]))
                    if vertical
                    else (max(p[0], q[0]), min(p[1], q[1]))
                )
                if hi - lo >= EDGE_MIN:
                    edges.append(Edge(min(i, j), max(i, j), lid, lo, hi))
    return edges


def _root(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _tree(cells: list[Cell], edges: list[Edge], geom: np.random.Generator) -> list[int]:
    """Randomised Kruskal over the shared walls, first those inside districts, then those across district
    walls, so the districts join through as few openings as the tree needs."""
    parent = list(range(len(cells)))
    tree: list[int] = []
    inside = [k for k, e in enumerate(edges) if cells[e.a].district == cells[e.b].district]
    across = [k for k, e in enumerate(edges) if cells[e.a].district != cells[e.b].district]
    for group in (inside, across):
        for idx in geom.permutation(len(group)):
            k = group[int(idx)]
            ra, rb = _root(parent, edges[k].a), _root(parent, edges[k].b)
            if ra != rb:
                parent[ra] = rb
                tree.append(k)
    if len(tree) != len(cells) - 1:
        raise Infeasible("cells_not_connected")
    return sorted(tree)


def _extra(
    cells: list[Cell],
    edges: list[Edge],
    tree: list[int],
    mixes: list[float],
    geom: np.random.Generator,
) -> list[int]:
    """Loops beyond the tree: per district round(braid · cells) of its inner shared walls, and one shared wall
    for every pair of neighbouring districts that has one the tree did not use."""
    in_tree = set(tree)
    inside: dict[int, list[int]] = {}
    across: dict[tuple[int, int], list[int]] = {}
    for k, e in enumerate(edges):
        if k in in_tree:
            continue
        da, db = cells[e.a].district, cells[e.b].district
        if da == db:
            inside.setdefault(da, []).append(k)
        else:
            across.setdefault((min(da, db), max(da, db)), []).append(k)
    counts = Counter(c.district for c in cells)
    extra: list[int] = []
    for d in sorted(inside):
        pool = inside[d]
        k = min(len(pool), round(_mix(BRAID, mixes[d]) * counts[d]))
        extra += [pool[int(p)] for p in geom.permutation(len(pool))[:k]]
    for pair in sorted(across):
        pool = across[pair]
        extra.append(pool[int(geom.integers(len(pool)))])
    return sorted(extra)


def _openings(
    cells: list[Cell],
    edges: list[Edge],
    opened: list[int],
    mixes: list[float],
    geom: np.random.Generator,
) -> list[Opening]:
    out: list[Opening] = []
    for k in opened:
        e = edges[k]
        ca, cb = cells[e.a], cells[e.b]
        room = e.hi - e.lo - 2 * MARGIN
        r = float(geom.random())
        kind: Kind
        if ca.district != cb.district:
            kind = "door" if r < P_DOOR_DISTRICT else "wide"
        else:
            mix = mixes[ca.district]
            if ca.corridor and cb.corridor:
                kind = "full" if r < _mix(P_FULL_CORRIDOR, mix) else "wide"
            elif ca.corridor or cb.corridor:
                kind = "door" if r < _mix(P_DOOR_CORRIDOR, mix) else "wide"
            else:
                p_door = _mix(P_DOOR_ROOM, mix)
                kind = (
                    "door" if r < p_door else "wide" if r < p_door + 0.6 * (1 - p_door) else "full"
                )
        if kind == "door":
            width = int(geom.integers(DOOR[0], DOOR[1] + 1))
        elif kind == "wide":
            width = int(geom.integers(WIDE[0], WIDE[1] + 1))
        else:
            width = room
        if width >= room:
            out.append(Opening(k, e.lo, e.hi - e.lo, "full"))
            continue
        start = e.lo + MARGIN + int(geom.integers(0, room - width + 1))
        out.append(Opening(k, start, width, kind))
    return out


def _pieces(lines: list[Line], edges: list[Edge], openings: list[Opening]) -> Pieces:
    """The solid stretches of every line, along it, after the openings are cut."""
    cuts: dict[int, list[tuple[int, int]]] = {}
    for o in openings:
        cuts.setdefault(edges[o.edge].line, []).append((o.start, o.start + o.width))
    pieces: list[tuple[tuple[int, int], ...]] = []
    for lid, line in enumerate(lines):
        cur, end = line.start - line.ext0, line.end + line.ext1
        out: list[tuple[int, int]] = []
        for a, b in sorted(cuts.get(lid, [])):
            if a > cur:
                out.append((cur, a))
            cur = b
        if end > cur:
            out.append((cur, end))
        pieces.append(tuple(out))
    return tuple(pieces)


def _gap(p: Rect, q: Rect) -> float:
    """Distance between two rectangles (0 when they touch or overlap)."""
    dx = max(0, p[0] - q[1], q[0] - p[1])
    dy = max(0, p[2] - q[3], q[2] - p[3])
    return math.hypot(dx, dy)


_OUTSIDE: dict[str, Rect] = {
    "w": (-BOUNDARY_T, 0, -BOUNDARY_T, N + BOUNDARY_T),
    "e": (N, N + BOUNDARY_T, -BOUNDARY_T, N + BOUNDARY_T),
    "s": (-BOUNDARY_T, N + BOUNDARY_T, -BOUNDARY_T, 0),
    "n": (-BOUNDARY_T, N + BOUNDARY_T, N, N + BOUNDARY_T),
}


def _side_rects(node: Node, lines: list[Line], pieces: Pieces) -> list[tuple[str, int, Rect]]:
    """The wall pieces around a cell, within CLEAR of its extent: (side, piece index or -1, rectangle)."""
    out: list[tuple[str, int, Rect]] = []
    for side, lid in (("w", node.w), ("e", node.e), ("s", node.s), ("n", node.n)):
        if lid == BOUNDARY:
            out.append((side, -1, _OUTSIDE[side]))
            continue
        line = lines[lid]
        lo, hi = (node.y0, node.y1) if line.axis == "v" else (node.x0, node.x1)
        for k, (a, b) in enumerate(pieces[lid]):
            if b <= lo - CLEAR or a >= hi + CLEAR:
                continue
            c0, c1 = line.c - line.half, line.c + line.half
            out.append((side, k, (c0, c1, a, b) if line.axis == "v" else (a, b, c0, c1)))
    return out


def _stub(
    node: Node, inner: Rect, pieces: Pieces, rng: np.random.Generator
) -> tuple[Rect, str, int] | None:
    """One stub candidate: a wall from a cell side into the room, (rectangle, side, piece index)."""
    ix0, ix1, iy0, iy1 = inner
    side = ("w", "e", "s", "n")[int(rng.integers(4))]
    along_x = side in ("s", "n")
    lo, hi = (ix0, ix1) if along_x else (iy0, iy1)
    if hi - lo <= 2 * CLEAR:
        return None
    t = int(rng.integers(lo + CLEAR, hi - CLEAR))
    depth = (iy1 - iy0) if along_x else (ix1 - ix0)
    length = int(rng.integers(STUB_LEN[0], max(STUB_LEN[0], int(STUB_LEN[1] * depth)) + 1))
    if length > depth - CLEAR:
        return None
    t0, t1 = t - WALL_HALF, t + WALL_HALF
    rect: Rect = {
        "w": (ix0, ix0 + length, t0, t1),
        "e": (ix1 - length, ix1, t0, t1),
        "s": (t0, t1, iy0, iy0 + length),
        "n": (t0, t1, iy1 - length, iy1),
    }[side]
    lid = getattr(node, side)
    if lid == BOUNDARY:
        return rect, side, -1
    for k, (a, b) in enumerate(pieces[lid]):
        # the base lies on one solid piece, at least CLEAR from both of its ends (openings or line ends)
        if a + CLEAR <= t0 and t1 <= b - CLEAR:
            return rect, side, k
    return None


def _place_stub(
    node: Node,
    inner: Rect,
    sides: list[tuple[str, int, Rect]],
    pieces: Pieces,
    placed: list[Rect],
    rng: np.random.Generator,
    attempts: int,
) -> Rect | None:
    """A stub at least CLEAR from every wall piece but its own and from the room's other stubs and partitions,
    or None when `attempts` candidates all fail."""
    for _ in range(attempts):
        found = _stub(node, inner, pieces, rng)
        if found is None:
            continue
        rect, side, piece = found
        others = [r for s, k, r in sides if not (s == side and k == piece)]
        if all(_gap(rect, r) >= CLEAR for r in others + placed):
            return rect
    return None


def _furnish(
    node: Node, mix: float, lines: list[Line], pieces: Pieces, rng: np.random.Generator
) -> tuple[list[Rect], list[Rect]]:
    """Wall stubs and free-standing partitions of one room."""
    inner = _interior(node, lines)
    ix0, ix1, iy0, iy1 = inner
    w, h = ix1 - ix0, iy1 - iy0
    per_100 = w * h / 10000.0  # (0.1 m)² to hundreds of m²
    sides = _side_rects(node, lines, pieces)
    placed: list[Rect] = []
    stubs: list[Rect] = []
    partitions: list[Rect] = []
    for _ in range(min(STUBS_MAX, int(rng.poisson(_mix(STUBS, mix) * per_100)))):
        rect = _place_stub(node, inner, sides, pieces, placed, rng, ATTEMPTS)
        if rect is not None:
            placed.append(rect)
            stubs.append(rect)
    for _ in range(min(PARTITIONS_MAX, int(rng.poisson(_mix(PARTITIONS, mix) * per_100)))):
        for _attempt in range(ATTEMPTS):
            along_x = rng.random() < 0.5
            span = w if along_x else h
            top = min(PARTITION_LEN[1], int(0.6 * span))
            if top < PARTITION_LEN[0]:
                break
            length = int(rng.integers(PARTITION_LEN[0], top + 1))
            xs, ys = (length, 2 * WALL_HALF) if along_x else (2 * WALL_HALF, length)
            if ix1 - CLEAR - xs <= ix0 + CLEAR or iy1 - CLEAR - ys <= iy0 + CLEAR:
                continue
            x = int(rng.integers(ix0 + CLEAR, ix1 - CLEAR - xs))
            y = int(rng.integers(iy0 + CLEAR, iy1 - CLEAR - ys))
            rect = (x, x + xs, y, y + ys)
            if all(_gap(rect, r) >= CLEAR for r in [r for _, _, r in sides] + placed):
                placed.append(rect)
                partitions.append(rect)
                break
    return stubs, partitions


def _rasterise(
    lines: list[Line], edges: list[Edge], openings: list[Opening]
) -> tuple[BoolArray, BoolArray]:
    """District walls and cell walls on the 0.1 m grid, with the openings cut."""
    main = np.zeros((N, N), dtype=bool)
    wall = np.zeros((N, N), dtype=bool)
    for line in lines:
        grid = main if line.main else wall
        a0, a1 = max(0, line.start - line.ext0), min(N, line.end + line.ext1)
        c0, c1 = line.c - line.half, line.c + line.half
        if line.axis == "v":
            grid[a0:a1, c0:c1] = True
        else:
            grid[c0:c1, a0:a1] = True
    for o in openings:
        line = lines[edges[o.edge].line]
        grid = main if line.main else wall
        c0, c1 = line.c - line.half, line.c + line.half
        if line.axis == "v":
            grid[o.start : o.start + o.width, c0:c1] = False
        else:
            grid[c0:c1, o.start : o.start + o.width] = False
    return main, wall


def _signature(window: BoolArray) -> str:
    """A hash of a cell's occupancy on 1 m blocks (a block is occupied when any of its 0.1 m cells is), the same
    for all eight rotations and mirror images: each image is blocked from its own corner. Cells that differ on 1 m
    blocks differ by more than a metre somewhere, and so also differ on the 0.1 m grid."""
    best = ""
    for k in range(4):
        turned = np.rot90(window, k)
        for view in (turned, turned.T):
            h, w = view.shape
            grid = np.zeros((-(-h // BLOCK) * BLOCK, -(-w // BLOCK) * BLOCK), dtype=bool)
            grid[:h, :w] = view
            ny, nx = grid.shape[0] // BLOCK, grid.shape[1] // BLOCK
            blocks = grid.reshape(ny, BLOCK, nx, BLOCK).any(axis=(1, 3))
            digest = hashlib.sha256(
                np.asarray(blocks.shape, dtype=np.int64).tobytes() + np.packbits(blocks).tobytes()
            ).hexdigest()
            if not best or digest < best:
                best = digest
    return best


def _window(padded: BoolArray, cell: Cell, lines: list[Line]) -> BoolArray:
    """A cell's rectangle with its walls, from occupancy padded by MAIN_HALF (the outer wall drawn MAIN_HALF
    thick)."""
    nd = cell.node
    hw, he, hs, hn = (
        MAIN_HALF if lid == BOUNDARY else lines[lid].half for lid in (nd.w, nd.e, nd.s, nd.n)
    )
    window: BoolArray = padded[
        nd.y0 - hs + MAIN_HALF : nd.y1 + hn + MAIN_HALF,
        nd.x0 - hw + MAIN_HALF : nd.x1 + he + MAIN_HALF,
    ]
    return window


def _signatures(cells: tuple[Cell, ...], lines: list[Line], occupied: BoolArray) -> list[str]:
    padded = np.pad(occupied, MAIN_HALF, constant_values=True)
    return [_signature(_window(padded, c, lines)) for c in cells]


def _islands(occupied: BoolArray) -> tuple[npt.NDArray[np.int32], list[tuple[int, Slices]]]:
    """The 8-connected obstacle components that do not touch the grid's edge (where the outer wall's network
    does), with their bounding slices."""
    labels, _ = ndimage.label(occupied, structure=raster.EIGHT)
    rim = np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]])
    edge = set(np.unique(rim).tolist())
    islands = [
        (lab, sl)
        for lab, sl in enumerate(ndimage.find_objects(labels), start=1)
        if sl is not None and lab not in edge
    ]
    return labels, islands


def _span(sl: Slices) -> int:
    """The longer side of a bounding slice pair, in units."""
    return int(max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start))


def _drop_fragments(main: BoolArray, wall: BoolArray, partitions: list[Rect]) -> int:
    """Remove the free-standing wall pieces shorter than FRAGMENT_MAX that full openings leave around junctions
    (partitions stay); returns how many."""
    labels, islands = _islands(main | wall)
    keep = {int(labels[y0, x0]) for x0, _, y0, _ in partitions}
    dropped = 0
    for lab, sl in islands:
        if lab in keep or _span(sl) >= FRAGMENT_MAX:
            continue
        mask = labels[sl] == lab
        main[sl][mask] = False
        wall[sl][mask] = False
        dropped += 1
    return dropped


def _distinguish(
    cells: list[Cell],
    lines: list[Line],
    pieces: Pieces,
    main: BoolArray,
    wall: BoolArray,
    stubs: list[tuple[int, Rect]],
    partitions: list[tuple[int, Rect]],
    rng: np.random.Generator,
) -> int:
    """Give every cell that matches an earlier one on 1 m blocks a stub of its own, so no two cells look alike;
    returns the number of stubs added (appended to `stubs`)."""
    padded = np.pad(main | wall, MAIN_HALF, constant_values=True)
    seen: set[str] = set()
    added = 0
    for i, c in enumerate(cells):
        sig = _signature(_window(padded, c, lines))
        if sig in seen:
            inner = _interior(c.node, lines)
            sides = _side_rects(c.node, lines, pieces)
            placed = [r for j, r in stubs + partitions if j == i]
            rect = _place_stub(c.node, inner, sides, pieces, placed, rng, 5 * ATTEMPTS)
            if rect is None:
                raise Infeasible("indistinct_cell")
            x0, x1, y0, y1 = rect
            wall[y0:y1, x0:x1] = True
            padded[y0 + MAIN_HALF : y1 + MAIN_HALF, x0 + MAIN_HALF : x1 + MAIN_HALF] = True
            stubs.append((i, rect))
            added += 1
            sig = _signature(_window(padded, c, lines))
            if sig in seen:
                raise Infeasible("indistinct_cell")
        seen.add(sig)
    return added


def _layout(seed: int) -> Layout:
    geom, furnish, _ = streams(seed)
    lines: list[Line] = []
    districts = _districts(geom, lines)
    order = geom.permutation(len(districts))
    # stratified mixes, so the districts span office to maze evenly
    strata = [(i + float(geom.random())) / len(districts) for i in range(len(districts))]
    mixes = [strata[int(order[d])] for d in range(len(districts))]
    cells: list[Cell] = []
    for d, district in enumerate(districts):
        cells += [
            Cell(node, d, corridor) for node, corridor in _cells(district, mixes[d], geom, lines)
        ]
    edges = _edges(cells, lines)
    linked = {i for e in edges for i in (e.a, e.b)}
    if len(linked) != len(cells):
        raise Infeasible("isolated_cell")
    tree = _tree(cells, edges, geom)
    extra = _extra(cells, edges, tree, mixes, geom)
    openings = _openings(cells, edges, sorted(tree + extra), mixes, geom)
    pieces = _pieces(lines, edges, openings)
    main, wall = _rasterise(lines, edges, openings)
    stubs: list[tuple[int, Rect]] = []
    partitions: list[tuple[int, Rect]] = []
    for i, c in enumerate(cells):
        x0, x1, y0, y1 = _interior(c.node, lines)
        if c.corridor or min(x1 - x0, y1 - y0) < FURNISH_MIN:
            continue
        got_stubs, got_parts = _furnish(c.node, mixes[c.district], lines, pieces, furnish)
        stubs += [(i, r) for r in got_stubs]
        partitions += [(i, r) for r in got_parts]
    for _, (x0, x1, y0, y1) in stubs + partitions:
        wall[y0:y1, x0:x1] = True
    wall &= ~main
    fragments = _drop_fragments(main, wall, [r for _, r in partitions])
    distinguishing = _distinguish(cells, lines, pieces, main, wall, stubs, partitions, furnish)
    free = ~(main | wall)
    _, components = ndimage.label(free)
    return Layout(
        tuple(districts),
        tuple(mixes),
        tuple(lines),
        tuple(cells),
        tuple(edges),
        tuple(tree),
        tuple(extra),
        tuple(openings),
        pieces,
        tuple(stubs),
        tuple(partitions),
        distinguishing,
        fragments,
        main,
        wall,
        int(components),
        round(float(np.count_nonzero(free)) * UNIT * UNIT, 4),
    )


def _clearance(lay: Layout) -> float:
    """The smallest gap between a stub or partition and anything else in its room (units), recomputed from the
    finished layout: each stub touches exactly one wall piece, a partition none. Stubs on the two faces of one
    wall belong to different rooms and are not compared."""
    lines = list(lay.lines)
    best = math.inf
    furniture: dict[int, list[tuple[Rect, bool]]] = {}
    for cell, rect in lay.stubs:
        furniture.setdefault(cell, []).append((rect, True))
    for cell, rect in lay.partitions:
        furniture.setdefault(cell, []).append((rect, False))
    for cell, items in furniture.items():
        walls = [r for _, _, r in _side_rects(lay.cells[cell].node, lines, lay.pieces)]
        for i, (rect, is_stub) in enumerate(items):
            wall_gaps = [_gap(rect, r) for r in walls]
            if sum(1 for g in wall_gaps if g == 0) != (1 if is_stub else 0):
                return 0.0
            gaps = [g for g in wall_gaps if g > 0]
            gaps += [_gap(rect, other) for j, (other, _) in enumerate(items) if j != i]
            best = min([best, *gaps])
    return best


def _spawns(free: BoolArray, prims: tuple[Primitive, ...]) -> tuple[Spawn, ...]:
    """S1 at the free cell with 2 m clearance nearest the centre, S2 the farthest such cell from S1 (a corner),
    S3 the farthest from S2 (the opposite corner), on a 1 m lattice of 0.1 m cell centres; each faces the longest
    of the four axis rays."""
    # padded with occupied cells: the outer wall lies just outside the grid; the distance runs between cell
    # centres, so a wall face is half a cell nearer than it
    dist = ndimage.distance_transform_edt(np.pad(free, 1))[1:-1, 1:-1] * UNIT
    rows, cols = np.nonzero(dist[5::10, 5::10] >= SPAWN_CLEAR_M + UNIT / 2)
    xs = -HALF_M + (cols * 10 + 5.5) * UNIT
    ys = -HALF_M + (rows * 10 + 5.5) * UNIT
    i1 = int(np.argmin(xs**2 + ys**2))
    i2 = int(np.argmax(np.hypot(xs - xs[i1], ys - ys[i1])))
    i3 = int(np.argmax(np.hypot(xs - xs[i2], ys - ys[i2])))
    notes = (
        "free cell with 2 m clearance nearest the centre",
        "such cell farthest from S1",
        "such cell farthest from S2",
    )
    spawns = []
    for k, (i, note) in enumerate(zip((i1, i2, i3), notes, strict=True), start=1):
        x, y = round(float(xs[i]), 4), round(float(ys[i]), 4)
        spawns.append(
            Spawn(f"S{k}", x, y, _heading(prims, x, y), f"{note}, along the longest axis ray")
        )
    return tuple(spawns)


def _heading(prims: tuple[Primitive, ...], x: float, y: float) -> float:
    """`longest_ray_heading` on the full-height 0.05 m raster of a 24 m window around (x, y): the rays reach
    10 m, and `families.common.full_height_raster` would rasterise the whole world (10⁸ cells)."""
    r = HEADING_WINDOW_M
    near = tuple(
        p
        for p in prims
        if (b := p.footprint_bounds())[0] < x + r and b[1] > x - r and b[2] < y + r and b[3] > y - r
    )
    grid = raster.Grid.around((x - r, x + r, y - r, y + r), margin=0.0)
    return longest_ray_heading(grid, raster.rasterise(grid, near, slab=None), x, y, AXIS_RAYS)


def _walls_disjoint(world: World) -> CheckResult:
    """`families.common.walls_check` counted on a 0.1 m raster: its pairwise test is quadratic in the 2,000-odd
    boxes. Wall boxes are disjoint and lie on the 0.1 m grid."""
    walls = [p for p in world.primitives if p.role in (Role.WALL, Role.BOUNDARY_WALL)]
    bounds = [p.footprint_bounds() for p in walls]
    gridded = all(abs(v * 10 - round(v * 10)) < 1e-6 for b in bounds for v in b)
    off = 10  # 1 m beyond the interior on every side
    counts = np.zeros((N + 2 * off, N + 2 * off), dtype=np.uint8)
    for b in bounds:
        c0, c1, r0, r1 = (round((v + HALF_M) * 10) + off for v in b)
        counts[r0:r1, c0:c1] += 1
    overlap = int(np.count_nonzero(counts > 1))
    return CheckResult(
        "LS1 wall boxes disjoint and on the 0.1 m grid",
        overlap == 0 and gridded,
        {"boxes": len(walls), "overlap_m2": round(overlap * UNIT * UNIT, 4), "on_grid": gridded},
    )


def _checks(lay: Layout) -> ExtraCheck:
    def run(world: World, a: Analysis) -> list[CheckResult]:
        parent = list(range(len(lay.cells)))
        for k in lay.tree + lay.extra:
            e = lay.edges[k]
            ra, rb = _root(parent, e.a), _root(parent, e.b)
            if ra != rb:
                parent[ra] = rb
        groups = len({_root(parent, i) for i in range(len(lay.cells))})
        disjoint = not set(lay.tree) & set(lay.extra)
        widths = [o.width for o in lay.openings]
        inside = all(
            (o.start, o.width)
            == (lay.edges[o.edge].lo, lay.edges[o.edge].hi - lay.edges[o.edge].lo)
            if o.kind == "full"
            else lay.edges[o.edge].lo + MARGIN <= o.start
            and o.start + o.width <= lay.edges[o.edge].hi - MARGIN
            for o in lay.openings
        )
        clear = _clearance(lay)
        occupied = lay.main | lay.wall
        signatures = _signatures(lay.cells, list(lay.lines), occupied)
        _, islands = _islands(occupied)
        fragments = sum(1 for _, sl in islands if _span(sl) < FRAGMENT_MAX)
        return [
            CheckResult(
                "LS1 cell graph: every cell reachable, openings = cells − 1 + extra",
                groups == 1 and disjoint and len(lay.tree) == len(lay.cells) - 1,
                {
                    "cells": len(lay.cells),
                    "shared_walls": len(lay.edges),
                    "tree": len(lay.tree),
                    "extra": len(lay.extra),
                },
            ),
            CheckResult(
                "LS1 openings at least 3.0 m wide and inside their shared walls",
                min(widths) >= DOOR[0] and inside,
                {
                    "openings": len(widths),
                    "min_m": _m(min(widths)),
                    "kinds": dict(sorted(Counter(o.kind for o in lay.openings).items())),
                },
            ),
            CheckResult(
                "LS1 no two cells alike on 1 m blocks (rotations and mirror images included)",
                len(set(signatures)) == len(signatures),
                {"cells": len(signatures), "distinct": len(set(signatures))},
            ),
            CheckResult(
                "LS1 no free-standing wall piece shorter than 2.5 m",
                fragments == 0,
                {"free_standing_pieces": len(islands), "shorter_than_2_5_m": fragments},
            ),
            CheckResult(
                "LS1 stubs and partitions at least 3.5 m from everything else in their room",
                clear >= CLEAR,
                None if math.isinf(clear) else _m(clear),
            ),
            CheckResult(
                "LS1 free space one 4-connected component on the construction grid",
                lay.components == 1,
                lay.components,
            ),
            CheckResult(
                "LS1 raster free area equals the construction",
                abs(a.free_area_m2 - lay.area_m2) < 0.005,
                {"raster_m2": round(a.free_area_m2, 2), "construction_m2": lay.area_m2},
            ),
            _walls_disjoint(world),
            no_symmetry("LS1 layout has no symmetry", world),
        ]

    return run


@dataclass(frozen=True, slots=True)
class Candidate:
    world: World
    layout: Layout


def _candidate(seed: int, candidate: int, rejected: list[dict[str, object]]) -> Candidate:
    lay = _layout(seed)
    outer, _ = walled_rectangle(
        -HALF_M, HALF_M, -HALF_M, HALF_M, thickness=_m(BOUNDARY_T), height=BOUNDARY_H_M
    )
    origin = (-HALF_M, -HALF_M)
    mains = grid_walls(
        lay.main, UNIT, origin=origin, prefix="main", height=WALL_H_M, material="core"
    )
    walls = grid_walls(lay.wall, UNIT, origin=origin, prefix="wall", height=WALL_H_M)
    prims = (*outer, *mains, *walls)
    spawns = _spawns(~(lay.main | lay.wall), prims)
    widths = sorted(o.width for o in lay.openings)
    corridors = sum(1 for c in lay.cells if c.corridor)
    world = World(
        id=WORLD_ID,
        slug="gargantua",
        title="Gargantua",
        tier="LS",
        role="evaluation",
        primitives=prims,
        spawns=spawns,
        expected=Expected(
            free_area_m2=round(lay.area_m2, 2),
            free_area_tolerance=0.005,
            # the narrowest openings are doors; every other gap is wider by construction
            narrowest_passage_m=_m(widths[0]),
            minimum_passage_m=_m(DOOR[0]),
        ),
        parameters={
            "interior_m": _m(N),
            "district_max_m": _m(DISTRICT_MAX),
            "cell_wall_m": _m(2 * WALL_HALF),
            "district_wall_m": _m(2 * MAIN_HALF),
            "boundary_wall_m": _m(BOUNDARY_T),
            "wall_height_m": WALL_H_M,
            "boundary_height_m": BOUNDARY_H_M,
            "door_width_m": [_m(DOOR[0]), _m(DOOR[1])],
            "wide_opening_m": [_m(WIDE[0]), _m(WIDE[1])],
            "corridor_spacing_m": [_m(STRIP[0]), _m(STRIP[1])],
            "furnishing_clearance_m": _m(CLEAR),
        },
        seed=seed,
        dimensions=(
            Dimension((-HALF_M, -HALF_M - 1.2), (HALF_M, -HALF_M - 1.2), f"{_m(N):.1f} m"),
        ),
        inclusion=(((-HALF_M, -HALF_M), (HALF_M, -HALF_M), (HALF_M, HALF_M), (-HALF_M, HALF_M)),),
        sources=({"what": "maze generation algorithms", "cite": "bellot2021mazes"},),
        extra_checks=(_checks(lay),),
        plan_data={
            "selection": {"candidate": candidate, "rejected": list(rejected)},
            "districts": [
                {
                    "id": d,
                    "office_maze_mix": round(lay.mixes[d], 3),
                    "rect_m": [_m(v - N // 2) for v in (node.x0, node.x1, node.y0, node.y1)],
                    "cells": sum(1 for c in lay.cells if c.district == d),
                }
                for d, node in enumerate(lay.districts)
            ],
            "cells": len(lay.cells),
            "corridor_cells": corridors,
            "room_cells": len(lay.cells) - corridors,
            "shared_walls": len(lay.edges),
            "tree_openings": len(lay.tree),
            "extra_openings": len(lay.extra),
            "openings_by_kind": dict(sorted(Counter(o.kind for o in lay.openings).items())),
            "opening_width_m": {
                "min": _m(widths[0]),
                "median": _m(statistics.median(widths)),
                "max": _m(widths[-1]),
            },
            "stubs": len(lay.stubs) - lay.distinguishing,
            "distinguishing_stubs": lay.distinguishing,
            "partitions": len(lay.partitions),
            "fragments_removed": lay.fragments,
            "construction_free_area_m2": lay.area_m2,
        },
        plan_width_px=6400,
        plan_detail_m=PLAN_DETAIL_M,
        overview_elevation_deg=60.0,
    )
    return Candidate(world, lay)


def _reject_reason(c: Candidate) -> str | None:
    """Selection runs the construction checks only: T0's analysis of a candidate takes about an hour, so it runs
    once, in the build, on the accepted candidate, and a T0 failure stops the build instead of trying the next
    seed. Distinct cells are built in (`_distinguish` raises `Infeasible` otherwise)."""
    lay = c.layout
    if lay.components != 1:
        return f"free space in {lay.components} components"
    return None


@functools.cache
def instance() -> Candidate:
    """The first accepted candidate (seed SEED_BASE + i)."""
    return select(WORLD_ID, SEED_BASE, _candidate, _reject_reason)


def build() -> World:
    return instance().world
