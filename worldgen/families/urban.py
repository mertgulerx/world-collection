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

"""F5 Walled urban blocks, a generated family of four street grids.

The actual block side b_eff = (100 − (n_b + 1)·w)/n_b is not a multiple of 0.1 m for C6-02 and C6-03 (25.3333 and
17.3333 m), so the construction uses integer units of 0.0001 m, the precision the SDF writes (T0 §4). Block edges
are the plan's exact values (plan §3.4 step 1) rounded once to that unit; streets are then exactly w wide. Recess
centres lie on 0.1 m steps from the face's start corner, wall notch centres on the 0.1 m grid.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from fractions import Fraction
from types import MappingProxyType

import numpy as np

from ..checks import Analysis, analyse
from ..geometry import Recess, box, walled_rectangle
from ..world import (
    CheckResult,
    Dimension,
    Expected,
    ExtraCheck,
    FamilyInfo,
    Kind,
    Notch,
    Point,
    Primitive,
    Role,
    Spawn,
    World,
)
from .common import (
    Infeasible,
    base_reject,
    first_failure,
    seed_rules,
    seeded_spawn,
    select,
    snap,
    snap_units,
    spaced_centres,
    streams,
    walls_check,
)

FAMILY = "F5"
SEED_BASE = 600000  # setting j tries seeds SEED_BASE + 1000·j + i (plan §3.3)
UNIT = 1e-4  # construction unit, 0.0001 m
PER_M = 10_000  # units per metre
STEP = 1_000  # 0.1 m in units
INTERIOR = 100  # interior side (m), plan §3.2
HALF = INTERIOR * PER_M // 2
BUILDING_H = 6.0
WALL_H = 3.0
MONUMENT_H = 3.0
RECESS_W = 2 * PER_M  # façade recess 2.0 m wide
RECESS_D = 1 * PER_M  # and 1.0 m deep
RECESS_CORNER = 3.0  # recess centres at least 3.0 m from the face corners
RECESS_GAP = 4 * PER_M  # and 4.0 m apart
RECESS_DRAWS = 1000
MONUMENT_MARGIN = 3.0  # monuments at least 3.0 m inside the plaza
WALL_RECESS_W = 1.0
WALL_RECESS_D = 0.4
WALL_RECESSES = 10  # per wall
WALL_RECESS_SPAN = 48.0  # centres uniform in [−48, 48]
WALL_RECESS_GAP = 30  # 3.0 m in 0.1 m steps
WALL_RECESS_DRAWS = 1000  # per wall
FACES = ("s", "e", "n", "w")
WALLS = ("s", "e", "n", "w")
PLAN_DETAIL_M = 30.0
OVERVIEW_ELEVATION_DEG = 60.0  # plan §3.5: buildings hide 3.5 m of street instead of 6 m
NARROW_TOL = 0.1  # raster tolerance of the narrowest passage (m)
# setting j: street width w (m), nominal block b (m)
SETTINGS = {1: (6, 15), 2: (6, 30), 3: (12, 15), 4: (12, 30)}
# planning ranges of plan §4.2: free area (m²), about ±2 standard deviations of the landmark areas
RANGES = {
    1: (6085.0, 6140.0),
    2: (5540.0, 5585.0),
    3: (7930.0, 7975.0),
    4: (6945.0, 6975.0),
}
# Settings not built: the maintainer dropped C6-02, C6-03 and C6-04 (18 October 2026; report §9 decision 29).
DROPPED = frozenset({2, 3, 4})
BUILT = tuple(j for j in SETTINGS if j not in DROPPED)
FAMILY_INFO = FamilyInfo(
    instances=tuple(f"{FAMILY}-{j:02d}" for j in BUILT),
    columns=2,
    seeds=seed_rules(SEED_BASE),
)

Rect = tuple[int, int, int, int]  # x0, x1, y0, y1 in units from the world origin


@dataclass(frozen=True, slots=True)
class Monument:
    block: int
    sx: float
    sy: float
    yaw: float
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class Layout:
    street: int  # w in units
    n: int  # n_b
    b_eff: Fraction  # exact block side in units
    spans: tuple[tuple[int, int], ...]  # start, end of column q (and row r), west to east
    plazas: tuple[int, ...]  # removed block indices, ascending
    recesses: tuple[
        tuple[int, str, int], ...
    ]  # block, face, centre offset from the face's start corner
    monuments: tuple[Monument, ...]  # one per plaza, ascending block index
    # side: centres in 0.1 m steps, in draw order (read-only)
    wall_recesses: Mapping[str, tuple[int, ...]]

    def rect(self, index: int) -> Rect:
        return _block_rect(self.spans, index)

    def kept(self) -> list[int]:
        return [i for i in range(self.n * self.n) if i not in self.plazas]


def _block_rect(spans: tuple[tuple[int, int], ...], index: int) -> Rect:
    """Block slot `index` (row-major from the south-west, n_b = len(spans) per row) in units."""
    n = len(spans)
    (x0, x1), (y0, y1) = spans[index % n], spans[index // n]
    return x0, x1, y0, y1


def _plaza_count(n: int) -> int:
    """p = max(1, round(0.2·n_b²)) removed blocks (plan §3.4 step 2)."""
    return max(1, round(0.2 * n * n))


def _round_units(value: Fraction) -> int:
    """Nearest unit, halves up."""
    return math.floor(value + Fraction(1, 2))


def _grid(w_m: int, b_m: int) -> tuple[int, Fraction, tuple[tuple[int, int], ...]]:
    """Plan §3.4 step 1: n_b = round((100 − w)/(b + w)), b_eff = (100 − (n_b + 1)·w)/n_b; block q spans
    −50 + w + q·(b_eff + w) to that value + b_eff."""
    n = round((INTERIOR - w_m) / (b_m + w_m))
    w = w_m * PER_M
    b_eff = Fraction(INTERIOR * PER_M - (n + 1) * w, n)
    spans = []
    for q in range(n):
        start = -HALF + w + q * (b_eff + w)
        spans.append((_round_units(start), _round_units(start + b_eff)))
    return n, b_eff, tuple(spans)


def _face_length(lay: Layout, index: int, face: str) -> int:
    x0, x1, y0, y1 = lay.rect(index)
    return x1 - x0 if face in ("s", "n") else y1 - y0


def _layout(j: int, geom: np.random.Generator, land: np.random.Generator) -> Layout:
    w_m, b_m = SETTINGS[j]
    n, b_eff, spans = _grid(w_m, b_m)
    p = _plaza_count(n)
    plazas = tuple(sorted(int(v) for v in geom.choice(n * n, p, replace=False)))
    b_eff_m = float(b_eff) * UNIT
    recesses: list[tuple[int, str, int]] = []
    for index in range(n * n):
        if index in plazas:
            continue
        for face in FACES:
            k = int(land.integers(0, 3))
            centres: list[int] = []
            draws = 0
            while len(centres) < k:
                draws += 1
                if draws > RECESS_DRAWS:
                    raise Infeasible("placement_failed")
                c = snap_units(float(land.uniform(RECESS_CORNER, b_eff_m - RECESS_CORNER))) * STEP
                if all(abs(c - other) >= RECESS_GAP for other in centres):
                    centres.append(c)
            recesses += [(index, face, c) for c in centres]
    monuments = []
    for index in plazas:
        sx = snap(float(land.uniform(2.0, 4.0)))
        sy = snap(float(land.uniform(2.0, 4.0)))
        yaw = round(float(land.uniform(0.0, math.pi / 2)), 6)
        rho = math.hypot(sx, sy) / 2
        x0, x1, y0, y1 = (v * UNIT for v in _block_rect(spans, index))
        inset = MONUMENT_MARGIN + rho
        x = round(float(land.uniform(x0 + inset, x1 - inset)), 4)
        y = round(float(land.uniform(y0 + inset, y1 - inset)), 4)
        monuments.append(Monument(index, sx, sy, yaw, x, y))
    walls: dict[str, tuple[int, ...]] = {}
    for side in WALLS:
        found, _ = spaced_centres(
            land,
            WALL_RECESSES,
            -WALL_RECESS_SPAN,
            WALL_RECESS_SPAN,
            WALL_RECESS_GAP,
            WALL_RECESS_DRAWS,
        )
        walls[side] = tuple(found)
    return Layout(
        w_m * PER_M,
        n,
        b_eff,
        spans,
        plazas,
        tuple(recesses),
        tuple(monuments),
        MappingProxyType(walls),
    )


def _recess_rect(lay: Layout, index: int, face: str, centre: int) -> Rect:
    """The notch of a façade recess; `centre` is measured from the face's start corner (its west end on the
    south and north faces, its south end on the east and west faces)."""
    x0, x1, y0, y1 = lay.rect(index)
    a, b = centre - RECESS_W // 2, centre + RECESS_W // 2
    if face == "s":
        return x0 + a, x0 + b, y0, y0 + RECESS_D
    if face == "n":
        return x0 + a, x0 + b, y1 - RECESS_D, y1
    if face == "w":
        return x0, x0 + RECESS_D, y0 + a, y0 + b
    return x1 - RECESS_D, x1, y0 + a, y0 + b


def _cut(start: int, end: int, holes: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """The parts of [start, end] outside the holes."""
    parts, cursor = [], start
    for a, b in sorted(holes):
        if a > cursor:
            parts.append((cursor, a))
        cursor = max(cursor, b)
    if end > cursor:
        parts.append((cursor, end))
    return parts


def _block_rects(lay: Layout, index: int) -> list[tuple[str, Rect]]:
    """Plan §3.4 step 3: a block outline with its notches as disjoint boxes: a core inset by the recess depth,
    the south and north strips over the full width, and the west and east strips between them, each strip cut
    at its recesses (recess centres are at least 3.0 m from the corners, so every notch lies in one strip)."""
    x0, x1, y0, y1 = lay.rect(index)
    d = RECESS_D
    holes: dict[str, list[tuple[int, int]]] = {face: [] for face in FACES}
    for block, face, centre in lay.recesses:
        if block == index:
            rx0, rx1, ry0, ry1 = _recess_rect(lay, index, face, centre)
            holes[face].append((rx0, rx1) if face in ("s", "n") else (ry0, ry1))
    out: list[tuple[str, Rect]] = [("core", (x0 + d, x1 - d, y0 + d, y1 - d))]
    for face, ya, yb in (("s", y0, y0 + d), ("n", y1 - d, y1)):
        out += [
            (f"{face}_{i:02d}", (a, b, ya, yb))
            for i, (a, b) in enumerate(_cut(x0, x1, holes[face]))
        ]
    for face, xa, xb in (("w", x0, x0 + d), ("e", x1 - d, x1)):
        out += [
            (f"{face}_{i:02d}", (xa, xb, a, b))
            for i, (a, b) in enumerate(_cut(y0 + d, y1 - d, holes[face]))
        ]
    return out


def _m(v: int) -> float:
    return v / PER_M


def _poly(rect: Rect) -> tuple[Point, ...]:
    x0, x1, y0, y1 = (_m(v) for v in rect)
    return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def _inclusion(walls: Mapping[str, tuple[int, ...]]) -> tuple[Point, ...]:
    """Plan §3.4 step 8: the interior square plus the wall notches, counter-clockwise from the south-west."""
    h, d, half_w = INTERIOR / 2, WALL_RECESS_D, WALL_RECESS_W / 2
    pts: list[Point] = [(-h, -h)]
    for c in sorted(v / 10 for v in walls["s"]):
        pts += [(c - half_w, -h), (c - half_w, -h - d), (c + half_w, -h - d), (c + half_w, -h)]
    pts.append((h, -h))
    for c in sorted(v / 10 for v in walls["e"]):
        pts += [(h, c - half_w), (h + d, c - half_w), (h + d, c + half_w), (h, c + half_w)]
    pts.append((h, h))
    for c in sorted((v / 10 for v in walls["n"]), reverse=True):
        pts += [(c + half_w, h), (c + half_w, h + d), (c - half_w, h + d), (c - half_w, h)]
    pts.append((-h, h))
    for c in sorted((v / 10 for v in walls["w"]), reverse=True):
        pts += [(-h, c + half_w), (-h - d, c + half_w), (-h - d, c - half_w), (-h, c - half_w)]
    return tuple((round(x, 4), round(y, 4)) for x, y in pts)


@dataclass(frozen=True, slots=True)
class Candidate:
    world: World
    layout: Layout


def _candidate(j: int, seed: int, candidate: int, rejected: list[dict[str, object]]) -> Candidate:
    w_m, b_m = SETTINGS[j]
    geom, land, spawn = streams(seed)
    lay = _layout(j, geom, land)
    n, p = lay.n, len(lay.plazas)
    h = INTERIOR / 2
    outer, wall_notches = walled_rectangle(
        -h,
        h,
        -h,
        h,
        recesses={
            side: [
                Recess(f"{side.upper()}{i + 1:02d}", c / 10, WALL_RECESS_W, WALL_RECESS_D)
                for i, c in enumerate(sorted(centres))
            ]
            for side, centres in lay.wall_recesses.items()
        },
        height=WALL_H,
    )
    buildings = [
        box(
            f"block_{index:02d}_{part}",
            _m(x0),
            _m(x1),
            _m(y0),
            _m(y1),
            BUILDING_H,
            Role.OBSTACLE,
            "inner",
        )
        for index in lay.kept()
        for part, (x0, x1, y0, y1) in _block_rects(lay, index)
    ]
    monuments = [
        Primitive(
            Kind.BOX,
            f"monument_{m.block:02d}",
            m.x,
            m.y,
            m.yaw,
            (m.sx, m.sy, MONUMENT_H),
            Role.LANDMARK,
            "landmark",
        )
        for m in lay.monuments
    ]
    facade_notches = []
    per_face: dict[tuple[int, str], int] = {}
    for index, face, centre in lay.recesses:
        k = per_face[(index, face)] = per_face.get((index, face), 0) + 1
        rect = _recess_rect(lay, index, face, centre)
        facade_notches.append(Notch(f"F{index:02d}{face.upper()}{k}", _poly(rect)))
    prims = (*outer, *buildings, *monuments)
    sx, sy, yaw = seeded_spawn(prims, (-h + w_m / 2, 0.0), spawn)
    block_area = sum(
        (x1 - x0) * (y1 - y0) for x0, x1, y0, y1 in (lay.rect(i) for i in lay.kept())
    ) / (PER_M * PER_M)
    recess_area = len(lay.recesses) * (RECESS_W * RECESS_D) / (PER_M * PER_M)
    wall_area = WALL_RECESS_W * WALL_RECESS_D * sum(len(c) for c in lay.wall_recesses.values())
    monument_area = sum(m.sx * m.sy for m in lay.monuments)
    area = round(INTERIOR * INTERIOR - block_area + recess_area + wall_area - monument_area, 4)
    loops = n * n
    b_eff_m = float(lay.b_eff) * UNIT
    x0, x1, y0, y1 = (_m(v) for v in lay.rect(0))
    xs1 = _m(lay.spans[1][0]) if n > 1 else h
    ym = (y0 + y1) / 2
    world = World(
        id=f"{FAMILY}-{j:02d}",
        slug=f"urban_w{w_m:02d}_b{b_m}",
        title=f"Walled urban blocks, {w_m} m streets, {b_m} m nominal blocks",
        tier="F",
        role="evaluation",
        primitives=prims,
        spawns=(Spawn("S1", sx, sy, yaw, "seeded free cell, along the longest open axis ray"),),
        expected=Expected(
            free_area_m2=round(area, 2),
            narrowest_passage_m=float(w_m),
            # opposite a 1.0 m wall notch the raster skeleton reads up to 0.06 m short (maintainer, 18 October
            # 2026): narrowest w ± 0.1 m, minimum w − 0.1 m
            narrowest_tolerance_m=NARROW_TOL,
            loops=loops,
            loops_unsized=loops,
            minimum_passage_m=round(w_m - NARROW_TOL, 4),
        ),
        family=FAMILY,
        parameters={
            "street_width_m": float(w_m),
            "nominal_block_m": float(b_m),
            "blocks_per_side": n,
            "block_side_m": round(b_eff_m, 4),
            "plazas": p,
            "building_height_m": BUILDING_H,
            "wall_height_m": WALL_H,
        },
        seed=seed,
        notches=(*wall_notches, *facade_notches),
        dimensions=(
            Dimension((-h, -h - 0.9), (h, -h - 0.9), f"{INTERIOR:.1f} m"),
            Dimension((x0, y0 - w_m / 2), (x1, y0 - w_m / 2), f"b_eff {b_eff_m:.2f} m"),
            Dimension((-h, ym), (x0, ym), f"w {w_m:.1f} m"),
            Dimension((x1, ym), (xs1, ym), f"w {w_m:.1f} m"),
        ),
        inclusion=(_inclusion(lay.wall_recesses),),
        sources=(
            {
                "what": "large campus scene for 3D-lidar exploration",
                "cite": "cao2022cmuexploration",
            },
            {"what": "feature-poor geometry bounds 2D localisation", "cite": "censi2007achievable"},
            {"what": "degeneracy of range-based localisation", "cite": "zhang2016degeneracy"},
        ),
        extra_checks=(_checks(j, lay),),
        plan_data={
            "selection": {"candidate": candidate, "rejected": list(rejected)},
            "block_spans_m": [[_m(a), _m(b)] for a, b in lay.spans],
            "plaza_blocks": list(lay.plazas),
            "facade_recesses": len(lay.recesses),
            "monuments": [
                {
                    "block": m.block,
                    "size_m": [m.sx, m.sy],
                    "yaw_rad": m.yaw,
                    "centre_m": [m.x, m.y],
                }
                for m in lay.monuments
            ],
            "wall_recesses": {
                s: [round(c / 10, 1) for c in sorted(v)] for s, v in lay.wall_recesses.items()
            },
            "construction_free_area_m2": area,
            "sheet_label": f"n_b {n}, b_eff {b_eff_m:.2f} m, {p} plaza{'s' if p > 1 else ''}, loops {loops}",
        },
        plan_width_px=3200,
        plan_detail_m=PLAN_DETAIL_M,
        overview_elevation_deg=OVERVIEW_ELEVATION_DEG,
    )
    return Candidate(world, lay)


def _reject_reason(j: int, c: Candidate, a: Analysis) -> str | None:
    """Why candidate `c` of setting j is rejected during selection, or None to accept it.

    Selection runs the shared rules of `common.base_reject` (passages of at least w − 0.1 m, loops n_b²) and the
    plan §7 family checks except the planning-range check, which only stops the build for review. The build
    runs everything else on the accepted candidate: `gz sdf -k`, the free-area tolerance, heights, determinism
    (T0 checks 1, 5, 6 and 8) and the planning-range check.
    """
    return base_reject(c.world, a) or first_failure(
        _family_checks(j, c.layout, c.world, a, ranges=False)
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


def _intersects(p: Primitive, x0: float, x1: float, y0: float, y1: float) -> bool:
    """Whether the footprint bounds of `p` overlap the open rectangle (by more than 1e-6 m)."""
    a = p.footprint_bounds()
    return min(a[1], x1) - max(a[0], x0) > 1e-6 and min(a[3], y1) - max(a[2], y0) > 1e-6


def _band_clear(a: Analysis, x0: float, x1: float, y0: float, y1: float) -> bool:
    """Every raster cell whose centre lies strictly inside the rectangle is reachable."""
    xs, ys = a.grid.centres()
    cols = np.nonzero((xs > x0 + 1e-6) & (xs < x1 - 1e-6))[0]
    rows = np.nonzero((ys > y0 + 1e-6) & (ys < y1 - 1e-6))[0]
    return bool(a.reachable[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].all())


def _family_checks(
    j: int, lay: Layout, world: World, a: Analysis, *, ranges: bool
) -> list[CheckResult]:
    """The family checks of plan §7 on one candidate; `ranges` adds the planning-range check."""
    n, w = lay.n, lay.street
    h = INTERIOR / 2
    results: list[CheckResult] = []
    edges = [-HALF] + [v for span in lay.spans for v in span] + [HALF]
    streets = [(edges[2 * k], edges[2 * k + 1]) for k in range(n + 1)]
    widths = sorted({_m(b - a_) for a_, b in streets})
    statics = world.static_primitives()
    bands = [(_m(a_), _m(b), -h, h) for a_, b in streets]
    bands += [(-h, h, _m(a_), _m(b)) for a_, b in streets]
    open_band = all(not _intersects(p, *band) for p in statics for band in bands)
    clear = all(_band_clear(a, *band) for band in bands)
    sides = sorted({_m(b - a_) for a_, b in lay.spans})
    results.append(
        CheckResult(
            "F5 grid: n_b² block slots; every street, the perimeter street included, w wide and clear",
            len(lay.spans) == n and all(b - a_ == w for a_, b in streets) and open_band and clear,
            {
                "blocks_per_side": n,
                "streets_m": widths,
                "block_sides_m": sides,
                "no_primitive_in_streets": open_band,
                "raster_clear": clear,
            },
        )
    )
    p = _plaza_count(n)
    obstacles = [q for q in statics if q.role is Role.OBSTACLE]
    landmarks = [q for q in statics if q.role is Role.LANDMARK]
    plaza_ok = len(lay.plazas) == p == len(set(lay.plazas)) == len(lay.monuments)
    min_inset = math.inf
    for index in lay.plazas:
        x0, x1, y0, y1 = (_m(v) for v in lay.rect(index))
        plaza_ok &= not any(_intersects(q, x0, x1, y0, y1) for q in obstacles)
        inside = [q for q in landmarks if _intersects(q, x0, x1, y0, y1)]
        plaza_ok &= len(inside) == 1
        for q in inside:
            bx0, bx1, by0, by1 = q.footprint_bounds()
            min_inset = min(min_inset, bx0 - x0, x1 - bx1, by0 - y0, y1 - by1)
    plaza_ok &= min_inset >= MONUMENT_MARGIN - 1e-4
    results.append(
        CheckResult(
            "F5 plazas: p removed blocks, one monument each, at least 3.0 m from the plaza edges",
            plaza_ok,
            {"plazas": list(lay.plazas), "min_inset_m": round(min_inset, 4)},
        )
    )
    counts = {(i, f): 0 for i in lay.kept() for f in FACES}
    for index, face, _ in lay.recesses:
        counts[(index, face)] += 1
    corner = [min(c, _face_length(lay, index, face) - c) * UNIT for index, face, c in lay.recesses]
    apart = [
        abs(c1 - c2) * UNIT
        for k, (i1, f1, c1) in enumerate(lay.recesses)
        for i2, f2, c2 in lay.recesses[k + 1 :]
        if (i1, f1) == (i2, f2)
    ]
    wall_gaps = [
        abs(u - v) / 10
        for centres in lay.wall_recesses.values()
        for k, u in enumerate(centres)
        for v in centres[k + 1 :]
    ]
    results.append(
        CheckResult(
            "F5 recesses: 0–2 per face, ≥ 3.0 m from corners, ≥ 4.0 m apart; wall notches ≥ 3.0 m apart",
            max(counts.values()) <= 2
            and min(corner, default=math.inf) >= RECESS_CORNER - 1e-9
            and min(apart, default=math.inf) >= RECESS_GAP * UNIT - 1e-9
            and all(len(c) == WALL_RECESSES for c in lay.wall_recesses.values())
            and min(wall_gaps) >= WALL_RECESS_GAP / 10 - 1e-9,
            {
                "facade_recesses": len(lay.recesses),
                "per_face": {str(k): sum(1 for v in counts.values() if v == k) for k in range(3)},
                "min_corner_m": round(min(corner, default=math.inf), 4),
                "min_apart_m": round(min(apart, default=math.inf), 4),
                "min_wall_gap_m": round(min(wall_gaps), 1),
            },
        )
    )
    results.append(
        CheckResult(
            "F5 loops: the T0 count equals n_b²",
            a.loops == n * n and a.loops_unsized == n * n,
            {"loops": a.loops, "loops_unsized": a.loops_unsized, "n_b2": n * n},
        )
    )
    walls = [q for q in statics if q.role is Role.BOUNDARY_WALL]
    heights_ok = (
        all(abs(q.height - BUILDING_H) < 1e-9 for q in obstacles)
        and all(abs(q.height - MONUMENT_H) < 1e-9 for q in landmarks)
        and all(abs(q.height - WALL_H) < 1e-9 for q in walls)
        and all(q.z_min == 0.0 for q in statics)
    )
    results.append(
        CheckResult(
            "F5 heights: buildings 6.0 m, monuments 3.0 m, walls 3.0 m, all from z = 0",
            heights_ok,
            {"buildings": len(obstacles), "monuments": len(landmarks), "walls": len(walls)},
        )
    )
    results.append(walls_check(FAMILY, world, (Role.BOUNDARY_WALL,)))
    overlap = sum(
        _intersects(q, *r.footprint_bounds())
        for k, q in enumerate(obstacles)
        for r in obstacles[k + 1 :]
    )
    on_units = all(
        abs(v * PER_M - round(v * PER_M)) < 1e-4 for q in obstacles for v in q.footprint_bounds()
    )
    results.append(
        CheckResult(
            "F5 building boxes disjoint and on the 0.0001 m construction grid",
            overlap == 0 and on_units,
            {"boxes": len(obstacles), "overlapping_pairs": overlap, "on_grid": on_units},
        )
    )
    if ranges:
        lo, hi = RANGES[j]
        area = round(a.free_area_m2, 2)
        results.append(
            CheckResult(
                "F5 free area inside the plan's planning range (outside: stop for review)",
                lo <= area <= hi,
                area,
                f"range {RANGES[j]}",
            )
        )
    return results


def _checks(j: int, lay: Layout) -> ExtraCheck:
    """The build's family checks, bound to this candidate's layout."""

    def run(world: World, a: Analysis) -> list[CheckResult]:
        return _family_checks(j, lay, world, a, ranges=True)

    return run


def build(j: int) -> World:
    return instance(j).world


def builders() -> dict[str, Callable[[], World]]:
    return {f"{FAMILY}-{j:02d}": functools.partial(build, j) for j in BUILT}
