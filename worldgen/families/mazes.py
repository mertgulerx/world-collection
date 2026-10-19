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

"""F1 Mazes, a generated family of eight grid mazes.

Lengths of the construction are integers in units of 0.1 m measured from the maze's south-west outer wall
centreline, so every wall lies on the 0.1 m construction grid (plan §3.2).
"""

from __future__ import annotations

import functools
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..checks import Analysis, analyse
from ..geometry import box, grid_walls, walled_rectangle
from ..world import (
    BoolArray,
    CheckResult,
    Dimension,
    Expected,
    ExtraCheck,
    FamilyInfo,
    Primitive,
    Role,
    Spawn,
    World,
)
from .common import (
    AXIS_RAYS,
    base_reject,
    first_failure,
    full_height_raster,
    longest_ray_heading,
    seed_rules,
    select,
    snap,
    streams,
    walls_check,
)

FAMILY = "F1"
SEED_BASE = 100000  # setting j tries seeds SEED_BASE + 1000·j + i (plan §3.3)
PITCH = 25  # cell pitch, 2.5 m in 0.1 m units
UNIT = 0.1
CLEAR = 1.15  # cell centre to wall face (m)
LANDMARK_DEPTH = 0.15
LANDMARK_WIDTHS = (0.3, 0.6, 1.0)
DIRECTIONS = ("e", "n", "w", "s")
STEP = {"e": (1, 0), "n": (0, 1), "w": (-1, 0), "s": (0, -1)}
# setting j: cells N, loops, algorithm
SETTINGS = {
    1: (8, "perfect", "rb"),
    2: (8, "perfect", "kr"),
    3: (8, "braid50", "rb"),
    4: (8, "braid50", "kr"),
    5: (14, "perfect", "rb"),
    6: (14, "perfect", "kr"),
    7: (14, "braid50", "rb"),
    8: (14, "braid50", "kr"),
}
# planning ranges of plan §4.2 (central 95 % of 1,000 draws): free area m², loops k, dead-end cells
RANGES = {
    1: ((366.2, 366.8), (0, 0), (5, 11)),
    2: ((366.2, 366.8), (0, 0), (15, 24)),
    3: ((366.7, 370.0), (1, 7), (1, 7)),
    4: ((368.5, 372.3), (5, 12), (2, 12)),
    5: ((1122.9, 1123.9), (0, 0), (17, 27)),
    6: ((1122.9, 1123.9), (0, 0), (52, 67)),
    7: ((1125.7, 1131.3), (6, 16), (5, 16)),
    8: ((1131.2, 1137.7), (18, 30), (11, 28)),
}
ALGORITHM_NAMES = {"rb": "recursive backtracker", "kr": "Kruskal"}
FAMILY_INFO = FamilyInfo(
    instances=tuple(f"{FAMILY}-{j:02d}" for j in SETTINGS),
    columns=4,
    # The recorded C1 text (worlds/C1_family/family.yaml) writes the family index: 100000·1.
    seeds={
        **seed_rules(SEED_BASE),
        "evaluation": "100000·1 + 1000·j + i (j: setting, i: candidate 0…999)",
    },
)

Edge = tuple[int, int]  # (smaller cell index, larger cell index)


@dataclass(frozen=True, slots=True)
class Maze:
    n: int
    open_edges: frozenset[Edge]
    braided: int  # k, walls opened by braiding

    def neighbours(self, cell: int) -> list[tuple[str, int | None]]:
        """(direction, neighbour or None at the boundary) in the fixed order east, north, west, south."""
        q, r = cell % self.n, cell // self.n
        out: list[tuple[str, int | None]] = []
        for d in DIRECTIONS:
            dq, dr = STEP[d]
            qq, rr = q + dq, r + dr
            out.append((d, rr * self.n + qq if 0 <= qq < self.n and 0 <= rr < self.n else None))
        return out

    def is_open(self, a: int, b: int) -> bool:
        return (min(a, b), max(a, b)) in self.open_edges

    def open_sides(self, cell: int) -> int:
        return sum(
            1 for _, nb in self.neighbours(cell) if nb is not None and self.is_open(cell, nb)
        )

    def dead_ends(self) -> int:
        return sum(1 for c in range(self.n * self.n) if self.open_sides(c) == 1)

    def open_squares(self) -> int:
        """u: interior vertices whose four incident edges are all open (2 × 2 cells with no wall).

        Each is a cell-graph cycle with no wall inside it, so the raster counts k − u loops (maintainer,
        18 October 2026).
        """
        n = self.n
        u = 0
        for i in range(1, n):
            for j in range(1, n):
                sw, se, nw, ne = (j - 1) * n + i - 1, (j - 1) * n + i, j * n + i - 1, j * n + i
                if all(self.is_open(a, b) for a, b in ((sw, se), (nw, ne), (sw, nw), (se, ne))):
                    u += 1
        return u

    @property
    def wall_loops(self) -> int:
        """Loops around a wall: k − u."""
        return self.braided - self.open_squares()


def _edge(a: int, b: int) -> Edge:
    return (min(a, b), max(a, b))


def _backtracker(n: int, geom: np.random.Generator) -> set[Edge]:
    start = int(geom.integers(n * n))
    probe = Maze(n, frozenset(), 0)
    visited = {start}
    stack = [start]
    edges: set[Edge] = set()
    while stack:
        top = stack[-1]
        options = [nb for _, nb in probe.neighbours(top) if nb is not None and nb not in visited]
        if not options:
            stack.pop()
            continue
        nb = options[int(geom.choice(len(options)))]
        edges.add(_edge(top, nb))
        visited.add(nb)
        stack.append(nb)
    return edges


def _kruskal(n: int, geom: np.random.Generator) -> set[Edge]:
    candidates: list[Edge] = []
    for cell in range(n * n):
        q, r = cell % n, cell // n
        if q < n - 1:
            candidates.append((cell, cell + 1))
        if r < n - 1:
            candidates.append((cell, cell + n))
    parent = list(range(n * n))

    def root(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    edges: set[Edge] = set()
    for idx in geom.permutation(len(candidates)):
        a, b = candidates[int(idx)]
        ra, rb = root(a), root(b)
        if ra != rb:
            parent[ra] = rb
            edges.add((a, b))
    return edges


def _braid(n: int, edges: set[Edge], geom: np.random.Generator) -> int:
    def maze() -> Maze:
        return Maze(n, frozenset(edges), 0)

    m = maze()
    ends = [c for c in range(n * n) if m.open_sides(c) == 1]
    opened = 0
    for idx in geom.permutation(len(ends)):
        cell = ends[int(idx)]
        m = maze()
        if m.open_sides(cell) != 1:
            continue
        if geom.random() >= 0.5:
            continue
        walled = [nb for _, nb in m.neighbours(cell) if nb is not None and not m.is_open(cell, nb)]
        dead = [nb for nb in walled if m.open_sides(nb) == 1]
        pool = dead or walled
        nb = pool[int(geom.choice(len(pool)))]
        edges.add(_edge(cell, nb))
        opened += 1
    return opened


def generate_maze(n: int, loops: str, algorithm: str, geom: np.random.Generator) -> Maze:
    edges = _backtracker(n, geom) if algorithm == "rb" else _kruskal(n, geom)
    k = _braid(n, edges, geom) if loops == "braid50" else 0
    return Maze(n, frozenset(edges), k)


def _occupancy(m: Maze) -> BoolArray:
    """Internal walls and touched vertex squares on the 0.1 m grid inside the interior (plan §3.4 step 5)."""
    n = m.n
    size = PITCH * n - 2
    occ = np.zeros((size, size), dtype=bool)
    for cell in range(n * n):
        q, r = cell % n, cell // n
        if q < n - 1 and not m.is_open(cell, cell + 1):
            x = PITCH * (q + 1)
            occ[PITCH * r : PITCH * r + 23, x - 2 : x] = True
        if r < n - 1 and not m.is_open(cell, cell + n):
            y = PITCH * (r + 1)
            occ[y - 2 : y, PITCH * q : PITCH * q + 23] = True
    for i in range(1, n):
        for j in range(1, n):
            below = (j - 1) * n + i - 1, (j - 1) * n + i
            above = j * n + i - 1, j * n + i
            left = (j - 1) * n + i - 1, j * n + i - 1
            right = (j - 1) * n + i, j * n + i
            if any(not m.is_open(a, b) for a, b in (below, above, left, right)):
                occ[PITCH * j - 2 : PITCH * j, PITCH * i - 2 : PITCH * i] = True
    return occ


def _landmarks(
    m: Maze, half: float, land: np.random.Generator
) -> list[tuple[Primitive, int, float]]:
    """Wall pilasters (plan §3.4 step 9): (primitive, cell, distance from its ends to the cell's side ends)."""
    n = m.n
    count = round(n * n / 6)
    out: list[tuple[Primitive, int, float]] = []
    for idx in land.permutation(n * n):
        if len(out) == count:
            break
        cell = int(idx)
        walled = [d for d, nb in m.neighbours(cell) if nb is None or not m.is_open(cell, nb)]
        if not walled:
            continue
        side = walled[int(land.choice(len(walled)))]
        w = LANDMARK_WIDTHS[int(land.integers(3))]
        span = 0.85 - w / 2
        offset = snap(float(land.uniform(-span, span)))
        q, r = cell % n, cell // n
        cx, cy = -half + 2.5 * (q + 0.5), -half + 2.5 * (r + 0.5)
        a, b = offset - w / 2, offset + w / 2
        inner, outer = CLEAR - LANDMARK_DEPTH, CLEAR
        if side == "e":
            x0, x1, y0, y1 = cx + inner, cx + outer, cy + a, cy + b
        elif side == "w":
            x0, x1, y0, y1 = cx - outer, cx - inner, cy + a, cy + b
        elif side == "n":
            x0, x1, y0, y1 = cx + a, cx + b, cy + inner, cy + outer
        else:
            x0, x1, y0, y1 = cx + a, cx + b, cy - outer, cy - inner
        name = f"landmark_{len(out):02d}"
        prim = box(name, x0, x1, y0, y1, 2.0, Role.LANDMARK, "landmark")
        out.append((prim, cell, round(CLEAR - max(abs(a), abs(b)), 6)))
    return out


@dataclass(frozen=True, slots=True)
class Candidate:
    world: World
    maze: Maze
    landmarks: tuple[tuple[Primitive, int, float], ...]


def _candidate(j: int, seed: int, candidate: int, rejected: list[dict[str, object]]) -> Candidate:
    n, loops, algorithm = SETTINGS[j]
    geom, land, spawn = streams(seed)
    maze = generate_maze(n, loops, algorithm, geom)
    side = PITCH * n * UNIT  # L, between outer wall centrelines
    half = side / 2
    inner_half = round(half - 0.1, 6)
    outer, _ = walled_rectangle(-inner_half, inner_half, -inner_half, inner_half)
    walls = grid_walls(_occupancy(maze), UNIT, origin=(-inner_half, -inner_half))
    marks = tuple(_landmarks(maze, half, land))
    prims = (*outer, *walls, *(p for p, _, _ in marks))
    cell = int(spawn.integers(n * n))
    sx = round(-half + 2.5 * (cell % n + 0.5), 6)
    sy = round(-half + 2.5 * (cell // n + 0.5), 6)
    yaw = longest_ray_heading(*full_height_raster(prims), sx, sy, AXIS_RAYS)
    blocked = sum(p.size[0] * p.size[1] for p in (*walls, *(p for p, _, _ in marks)))
    area = round((2 * inner_half) ** 2 - blocked, 4)
    instance = f"{FAMILY}-{j:02d}"
    slug = f"maze_s{n:02d}_{loops}_{algorithm}"
    loop_text = "perfect" if loops == "perfect" else "braid 0.5"
    world = World(
        id=instance,
        slug=slug,
        title=f"Maze {n}×{n}, {loop_text}, {ALGORITHM_NAMES[algorithm]}",
        tier="F",
        role="evaluation",
        primitives=prims,
        spawns=(Spawn("S1", sx, sy, yaw, f"centre of cell {cell}, along the longest open ray"),),
        expected=Expected(
            free_area_m2=round(area, 2),
            narrowest_passage_m=2.1,  # T0 reads 2.0–2.2 m (maintainer, 18 October 2026; plan §4.1)
            narrowest_tolerance_m=0.1,
            loops=maze.wall_loops,
            loops_unsized=maze.wall_loops,
            minimum_passage_m=2.0,
        ),
        family=FAMILY,
        parameters={
            "cells": n,
            "loops": loop_text,
            "algorithm": ALGORITHM_NAMES[algorithm],
            "cell_pitch_m": 2.5,
            "landmarks": len(marks),
        },
        seed=seed,
        dimensions=(
            Dimension((-half, -half - 0.9), (half, -half - 0.9), f"{side:.1f} m"),
            Dimension((-half, half + 0.5), (-half + 2.5, half + 0.5), "2.5 m"),
        ),
        inclusion=(
            (
                (-inner_half, -inner_half),
                (inner_half, -inner_half),
                (inner_half, inner_half),
                (-inner_half, inner_half),
            ),
        ),
        sources=({"what": "maze generation algorithms", "cite": "bellot2021mazes"},),
        extra_checks=(_checks(j, maze, marks),),
        plan_data={
            "selection": {"candidate": candidate, "rejected": list(rejected)},
            "braided_walls": maze.braided,
            "open_squares": maze.open_squares(),
            "dead_end_cells": maze.dead_ends(),
            "dead_end_fraction": round(maze.dead_ends() / (n * n), 4),
            "spawn_cell": cell,
            "construction_free_area_m2": area,
        },
    )
    return Candidate(world, maze, marks)


def _reject_reason(j: int, c: Candidate, a: Analysis) -> str | None:
    """Why candidate `c` of setting j is rejected during selection, or None to accept it.

    Selection runs the shared rules of `common.base_reject` (the loop rule compares the raster with k − u through
    `Expected`) and the plan §7 family checks except the planning-range check, which only stops the build for
    review. The build runs everything else on the accepted candidate: `gz sdf -k`, the free-area tolerance,
    heights, determinism (T0 checks 1, 5, 6 and 8) and the planning-range check.
    """
    return base_reject(c.world, a) or first_failure(
        _family_checks(j, c.maze, c.landmarks, c.world, a, ranges=False)
    )


@functools.cache
def instance(j: int) -> Candidate:
    """The first accepted candidate of setting j (seed SEED_BASE + 1000·j + i)."""
    return select(
        f"{FAMILY}-{j:02d}",
        SEED_BASE + 1000 * j,
        functools.partial(_candidate, j),
        lambda c: _reject_reason(j, c, analyse(c.world)),
    )


def _family_checks(
    j: int,
    m: Maze,
    landmarks: tuple[tuple[Primitive, int, float], ...],
    world: World,
    a: Analysis,
    *,
    ranges: bool,
) -> list[CheckResult]:
    """The family checks of plan §7 on one candidate; `ranges` adds the planning-range check."""
    n = m.n
    results: list[CheckResult] = []
    seen = {0}
    todo = [0]
    while todo:
        cell = todo.pop()
        for _, nb in m.neighbours(cell):
            if nb is not None and nb not in seen and m.is_open(cell, nb):
                seen.add(nb)
                todo.append(nb)
    edges = len(m.open_edges)
    results.append(
        CheckResult(
            "F1 cell graph: every cell reachable, open edges = N² − 1 + k (raster loops = k − u)",
            len(seen) == n * n and edges == n * n - 1 + m.braided,
            {
                "reachable": len(seen),
                "open_edges": edges,
                "k": m.braided,
                "open_squares_u": m.open_squares(),
            },
        )
    )
    results.append(
        CheckResult(
            "F1 narrowest passage (T0 reading) within 2.0–2.2 m",
            a.narrowest_m is not None and 2.0 - 1e-9 <= a.narrowest_m <= 2.2 + 1e-9,
            a.narrowest_m,
        )
    )
    cells = [cell for _, cell, _ in landmarks]
    ends = [end for _, _, end in landmarks]
    want = round(n * n / 6)
    results.append(
        CheckResult(
            "F1 landmarks: Λ = round(N²/6), at most one per cell, ends ≥ 0.3 m from the side ends",
            len(cells) == want and len(set(cells)) == len(cells) and min(ends) >= 0.3 - 1e-9,
            {"count": len(cells), "expected": want, "min_end_m": min(ends)},
        )
    )
    results.append(walls_check(FAMILY, world))
    dead = m.dead_ends()
    if ranges:
        (alo, ahi), (klo, khi), (dlo, dhi) = RANGES[j]
        area = round(a.free_area_m2, 2)
        inside = alo <= area <= ahi and klo <= m.braided <= khi and dlo <= dead <= dhi
        results.append(
            CheckResult(
                "F1 values inside the plan's planning ranges (outside: stop for review)",
                inside,
                {"free_area_m2": area, "k": m.braided, "dead_end_cells": dead},
                f"ranges {RANGES[j]}",
            )
        )
    results.append(
        CheckResult(
            "F1 dead-end fraction (recorded, not a rule)",
            True,
            round(dead / (n * n), 4),
        )
    )
    return results


def _checks(j: int, m: Maze, landmarks: tuple[tuple[Primitive, int, float], ...]) -> ExtraCheck:
    """The build's family checks, bound to this candidate's maze and landmarks."""

    def run(world: World, a: Analysis) -> list[CheckResult]:
        return _family_checks(j, m, landmarks, world, a, ranges=True)

    return run


def build(j: int) -> World:
    return instance(j).world


def builders() -> dict[str, Callable[[], World]]:
    return {f"{FAMILY}-{j:02d}": functools.partial(build, j) for j in SETTINGS}
