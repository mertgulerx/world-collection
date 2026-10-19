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

"""F4 Walled forests, a generated family of four walled squares of trunks.

Origin at the centre of the square, x east, y north (plan §3.2). Trunk radii and positions and boulder centres are
rounded to 0.1 mm when drawn, so every gap test runs on the values the SDF writes (T0 §4: 4 decimals for metres,
6 for radians). Wall recesses and boulder sides lie on the 0.1 m grid (plan §3.4).
"""

from __future__ import annotations

import functools
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from ..checks import Analysis, analyse
from ..geometry import Recess, cylinder, walled_rectangle
from ..world import (
    CheckResult,
    Dimension,
    Expected,
    ExtraCheck,
    FamilyInfo,
    Kind,
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
    spaced_centres,
    streams,
    walls_check,
)

FAMILY = "F4"
SEED_BASE = 500000  # setting j tries seeds SEED_BASE + 1000·j + i (plan §3.3)
GAP = 1.0  # minimum surface gap between trunks, boulders and walls (plan §3.2)
R_MIN, R_MAX = 0.15, 0.40  # trunk radius (m)
TRUNK_H = 4.0
BOULDER_H = 1.5
WALL_H = 3.0
RECESS_W = 1.0
RECESS_D = 0.4
RECESS_GAP = 3.0
RECESS_GAP_UNITS = 30  # RECESS_GAP in 0.1 m units
DRAWS = 1000  # per wall (recesses) and per boulder
TRUNK_DRAWS = (
    10000  # per trunk (maintainer, 18 October 2026: 1,000 rejected 12.3 % of setting-2 candidates)
)
# T0 check 4 (maintainer, 18 October 2026): the skeleton reading under-reads diagonal gaps between discs by up to
# 0.15 m, so a reading in [0.85, exact smallest gap + 0.1] passes; the exact gap check (≥ 1.0 m) is the
# authoritative passage rule.
T0_MINIMUM = 0.9  # minimum_passage_m; check 4 and the acceptance allow one raster cell below it
T0_ABOVE = 0.1  # allowance above the exact smallest gap
HASH_M = 2.0  # spatial-hash cell; r + r' + 1.0 ≤ 1.8 m, so the 3 × 3 neighbourhood suffices
DETAIL_M = 20.0
# setting j: side S (m), density λ (trunks per m²)
# Densities lowered to realistic stands (maintainer, 18 October 2026, decision 28): 0.01 and 0.03 trunks per m²,
# about 100 and 300 stems per hectare, basal area about 25 and 76 m²/ha with these radii (was 0.05 and 0.2).
SETTINGS = {1: (40, 0.01), 2: (40, 0.03), 3: (80, 0.01), 4: (80, 0.03)}
# planning ranges of plan §4.2: free area (m²), ±2 standard deviations of trunk and boulder area
RANGES = {1: (1583.0, 1591.0), 2: (1574.0, 1584.0), 3: (6328.0, 6345.0), 4: (6294.0, 6314.0)}
# Settings not built: the maintainer dropped setting 1 (18 October 2026; report §9 decision 29).
DROPPED = frozenset({1})
BUILT = tuple(j for j in SETTINGS if j not in DROPPED)


def world_id(j: int) -> str:
    """The instance ID of setting j: its position among the built settings, so IDs have no gaps (report §9
    decision 51). Seeds still follow the setting number j."""
    return f"{FAMILY}-{BUILT.index(j) + 1:02d}"


FAMILY_INFO = FamilyInfo(
    instances=tuple(world_id(j) for j in BUILT),
    columns=2,
    seeds=seed_rules(SEED_BASE),
)


@dataclass(frozen=True, slots=True)
class Boulder:
    x: float
    y: float
    long: float
    short: float
    yaw: float

    @property
    def rho(self) -> float:
        """Radius of the exclusion disc, the half-diagonal (plan §3.4 step 3)."""
        return math.hypot(self.long, self.short) / 2

    def corners(self) -> list[tuple[float, float]]:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        out = []
        for u, v in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
            du, dv = u * self.long / 2, v * self.short / 2
            out.append((self.x + du * c - dv * s, self.y + du * s + dv * c))
        return out


@dataclass(frozen=True, slots=True)
class Layout:
    side: int
    density: float
    recesses: Mapping[
        str, tuple[float, ...]
    ]  # "s", "e", "n", "w": centres along the wall (m), read-only
    boulders: tuple[Boulder, ...]
    poisson_n: int
    trunks: tuple[tuple[float, float, float], ...]  # x, y, r
    max_draws: int  # most position draws one trunk needed


def _recesses(side: int, land: np.random.Generator) -> Mapping[str, tuple[float, ...]]:
    """Plan §3.4 step 2: S/10 recesses per wall (south, east, north, west), centres ≥ 3.0 m apart; each wall
    has its own draw budget."""
    half = side / 2
    out: dict[str, tuple[float, ...]] = {}
    for wall in ("s", "e", "n", "w"):
        centres, _ = spaced_centres(
            land, side // 10, -half + 2.0, half - 2.0, RECESS_GAP_UNITS, DRAWS
        )
        out[wall] = tuple(c / 10 for c in centres)
    return MappingProxyType(out)


def _boulders(side: int, land: np.random.Generator) -> tuple[Boulder, ...]:
    """Plan §3.4 step 3: (S/20)² boxes; discs keep 1.0 m from the wall faces and from each other."""
    half = side / 2
    placed: list[Boulder] = []
    for _ in range((side // 20) ** 2):
        long = snap(float(land.uniform(2.0, 3.0)))
        short = snap(float(land.uniform(1.0, 2.0)))
        yaw = round(float(land.uniform(0.0, math.pi)), 6)
        rho = math.hypot(long, short) / 2
        lim = half - rho - GAP
        for _ in range(DRAWS):
            x = round(float(land.uniform(-lim, lim)), 4)
            y = round(float(land.uniform(-lim, lim)), 4)
            if all(math.hypot(x - b.x, y - b.y) >= rho + b.rho + GAP for b in placed):
                placed.append(Boulder(x, y, long, short, yaw))
                break
        else:
            raise Infeasible("placement_failed")
    return tuple(placed)


def _trunks(
    side: int, density: float, boulders: tuple[Boulder, ...], geom: np.random.Generator
) -> tuple[int, tuple[tuple[float, float, float], ...], int]:
    """Plan §3.4 step 4: n ~ Poisson(λS²) trunks by random sequential placement with a 1.0 m hard-core gap.

    The radius is drawn before the position draws; a 2 m spatial hash keeps the gap test local.
    """
    half = side / 2
    n = int(geom.poisson(density * side * side))
    trunks: list[tuple[float, float, float]] = []
    cells: dict[tuple[int, int], list[int]] = {}
    most = 0
    for _ in range(n):
        r = round(float(geom.uniform(R_MIN, R_MAX)), 4)
        lim = half - r - GAP
        for draw in range(1, TRUNK_DRAWS + 1):
            x = round(float(geom.uniform(-lim, lim)), 4)
            y = round(float(geom.uniform(-lim, lim)), 4)
            cx, cy = math.floor((x + half) / HASH_M), math.floor((y + half) / HASH_M)
            ok = all(math.hypot(x - b.x, y - b.y) - r - b.rho >= GAP for b in boulders) and all(
                math.hypot(x - trunks[k][0], y - trunks[k][1]) - r - trunks[k][2] >= GAP
                for i in (-1, 0, 1)
                for j in (-1, 0, 1)
                for k in cells.get((cx + i, cy + j), ())
            )
            if ok:
                cells.setdefault((cx, cy), []).append(len(trunks))
                trunks.append((x, y, r))
                most = max(most, draw)
                break
        else:
            raise Infeasible("placement_failed")
    return n, tuple(trunks), most


def _layout(j: int, seed: int) -> tuple[Layout, np.random.Generator]:
    """Plan §3.3: streams geom, land, spawn; landmarks before trunks."""
    side, density = SETTINGS[j]
    geom, land, spawn = streams(seed)
    recesses = _recesses(side, land)
    boulders = _boulders(side, land)
    n, trunks, most = _trunks(side, density, boulders, geom)
    return Layout(side, density, recesses, boulders, n, trunks, most), spawn


# --- exact gaps from the primitive list (plan §7) -----------------------------------------------------------


def _point_segment(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    ax, ay = a
    dx, dy = b[0] - ax, b[1] - ay
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    return math.hypot(p[0] - ax - t * dx, p[1] - ay - t * dy)


def _inside(p: tuple[float, float], b: Boulder) -> bool:
    c, s = math.cos(b.yaw), math.sin(b.yaw)
    u = (p[0] - b.x) * c + (p[1] - b.y) * s
    v = -(p[0] - b.x) * s + (p[1] - b.y) * c
    return abs(u) <= b.long / 2 and abs(v) <= b.short / 2


def _point_box(p: tuple[float, float], b: Boulder) -> float:
    """Distance from a point to a boulder's footprint (0 inside)."""
    if _inside(p, b):
        return 0.0
    k = b.corners()
    return min(_point_segment(p, k[i], k[(i + 1) % 4]) for i in range(4))


def _box_box(a: Boulder, b: Boulder) -> float:
    """Distance between two convex footprints (0 if they overlap)."""
    ka, kb = a.corners(), b.corners()
    if any(_inside(p, b) for p in ka) or any(_inside(p, a) for p in kb):
        return 0.0
    return min(
        *(_point_segment(p, kb[i], kb[(i + 1) % 4]) for p in ka for i in range(4)),
        *(_point_segment(p, ka[i], ka[(i + 1) % 4]) for p in kb for i in range(4)),
    )


@dataclass(frozen=True, slots=True)
class Gaps:
    trunk_trunk: float
    trunk_disc: float
    trunk_wall: float
    disc_disc: float
    disc_wall: float
    surface: (
        float  # smallest real surface gap between any two solids (trunk, boulder box, wall face)
    )
    nearest_median: float  # median over trunks of the gap to the nearest other trunk


def _gaps(lay: Layout) -> Gaps:
    half = lay.side / 2
    t = np.asarray(lay.trunks, dtype=np.float64).reshape(-1, 3)
    inf = math.inf
    if len(t) > 1:
        d = np.hypot(t[:, None, 0] - t[None, :, 0], t[:, None, 1] - t[None, :, 1])
        g = d - t[:, None, 2] - t[None, :, 2]
        np.fill_diagonal(g, inf)
        nearest = g.min(axis=1)
        tt, median = float(nearest.min()), float(np.median(nearest))
    else:
        tt, median = inf, inf
    tw = float((half - np.abs(t[:, :2]).max(axis=1) - t[:, 2]).min()) if len(t) else inf
    bs = lay.boulders
    td = min(
        (math.hypot(x - b.x, y - b.y) - r - b.rho for x, y, r in lay.trunks for b in bs),
        default=inf,
    )
    tb = min((_point_box((x, y), b) - r for x, y, r in lay.trunks for b in bs), default=inf)
    dd = min(
        (
            math.hypot(a.x - b.x, a.y - b.y) - a.rho - b.rho
            for i, a in enumerate(bs)
            for b in bs[i + 1 :]
        ),
        default=inf,
    )
    dw = min((half - max(abs(b.x), abs(b.y)) - b.rho for b in bs), default=inf)
    bw = min(
        (half - max(max(abs(x), abs(y)) for x, y in b.corners()) for b in bs),
        default=inf,
    )
    bb = min((_box_box(a, b) for i, a in enumerate(bs) for b in bs[i + 1 :]), default=inf)
    return Gaps(tt, td, tw, dd, dw, min(tt, tw, tb, bw, bb), median)


# --- candidate ----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Candidate:
    world: World
    layout: Layout
    gaps: Gaps


def _candidate(j: int, seed: int, candidate: int, rejected: list[dict[str, object]]) -> Candidate:
    lay, spawn = _layout(j, seed)
    side, density = SETTINGS[j]
    half = side / 2
    outer, notches = walled_rectangle(
        -half,
        half,
        -half,
        half,
        recesses={
            wall: [
                Recess(f"{wall.upper()}{i + 1}", c, RECESS_W, RECESS_D)
                for i, c in enumerate(sorted(centres))
            ]
            for wall, centres in lay.recesses.items()
        },
        height=WALL_H,
    )
    boulders = [
        Primitive(
            Kind.BOX,
            f"boulder_{i:02d}",
            b.x,
            b.y,
            b.yaw,
            (b.long, b.short, BOULDER_H),
            Role.LANDMARK,
            "landmark",
        )
        for i, b in enumerate(lay.boulders)
    ]
    trunks = [
        cylinder(f"trunk_{i:04d}", x, y, r, TRUNK_H, Role.OBSTACLE, "obstacle")
        for i, (x, y, r) in enumerate(lay.trunks)
    ]
    prims = (*outer, *boulders, *trunks)
    # (−S/2 + 0.5, 0) is free: trunks and boulder discs keep 1.0 m from the wall faces
    sx, sy, yaw = seeded_spawn(prims, (-half + 0.5, 0.0), spawn)
    n_rec = sum(len(c) for c in lay.recesses.values())
    trunk_area = sum(math.pi * r * r for _, _, r in lay.trunks)
    boulder_area = sum(b.long * b.short for b in lay.boulders)
    area = round(side * side + RECESS_W * RECESS_D * n_rec - boulder_area - trunk_area, 4)
    gaps = _gaps(lay)
    n = len(lay.trunks)
    achieved = round(n / (side * side), 6)
    width = 3200 if side == 80 else 1600
    # T0 reading window [T0_MINIMUM − h, exact + T0_ABOVE] as centre ± tolerance (h = 0.05 m)
    low, high = T0_MINIMUM - 0.05, round(gaps.surface, 2) + T0_ABOVE
    world = World(
        id=world_id(j),
        slug=f"forest_s{side}_d{round(density * 100):03d}",
        title=f"Walled forest {side}×{side} m, {density:g} trunks per m²",
        tier="F",
        role="evaluation",
        primitives=prims,
        spawns=(Spawn("S1", sx, sy, yaw, "seeded free cell, along the longest open axis ray"),),
        expected=Expected(
            free_area_m2=round(area, 2),
            narrowest_passage_m=round((low + high) / 2, 4),
            loops=len(lay.boulders),
            loops_unsized=n + len(lay.boulders),
            minimum_passage_m=T0_MINIMUM,
            narrowest_tolerance_m=round((high - low) / 2, 4),
        ),
        family=FAMILY,
        parameters={
            "side_m": side,
            "density_per_m2": density,
            "trunk_radius_m": [R_MIN, R_MAX],
            "trunk_height_m": TRUNK_H,
            "minimum_gap_m": GAP,
            "wall_height_m": WALL_H,
            "boulders": len(lay.boulders),
            "recesses_per_wall": side // 10,
        },
        seed=seed,
        notches=tuple(notches),
        dimensions=(
            Dimension((-half, -half - 0.9), (half, -half - 0.9), f"{side:.1f} m"),
            Dimension((-half - 0.9, -half), (-half - 0.9, half), f"{side:.1f} m"),
        ),
        sources=(
            {"what": "Poisson forest model", "cite": "karaman2012forest"},
            {"what": "trunk radius and density", "cite": "tian2020forest"},
            {"what": "localisation in feature-poor open space", "cite": "censi2007achievable"},
        ),
        extra_checks=(_checks(j, lay, gaps),),
        plan_width_px=width,
        plan_detail_m=DETAIL_M,
        plan_data={
            "selection": {"candidate": candidate, "rejected": list(rejected)},
            "poisson_draw": lay.poisson_n,
            "trunks": n,
            "achieved_density_per_m2": achieved,
            "max_draws_one_trunk": lay.max_draws,
            "holes_total": n + len(lay.boulders),
            "boulders": [
                {
                    "x": b.x,
                    "y": b.y,
                    "size_m": [b.long, b.short],
                    "yaw": b.yaw,
                    "disc_radius_m": round(b.rho, 4),
                }
                for b in lay.boulders
            ],
            "wall_recesses": {w: sorted(c) for w, c in lay.recesses.items()},
            "gaps_m": {
                "trunk_trunk": round(gaps.trunk_trunk, 4),
                "trunk_boulder_disc": round(gaps.trunk_disc, 4),
                "trunk_wall": round(gaps.trunk_wall, 4),
                "boulder_disc_disc": round(gaps.disc_disc, 4),
                "boulder_disc_wall": round(gaps.disc_wall, 4),
                "smallest_surface_gap": round(gaps.surface, 4),
                "median_nearest_trunk_gap": round(gaps.nearest_median, 4),
            },
            "trunk_area_m2": round(trunk_area, 4),
            "boulder_area_m2": round(boulder_area, 4),
            "construction_free_area_m2": area,
            "sheet_label": f"n {n}, density {achieved:.4f} per m²",
        },
    )
    return Candidate(world, lay, gaps)


def _reject_reason(j: int, c: Candidate, a: Analysis) -> str | None:
    """Why candidate `c` of setting j is rejected during selection, or None to accept it.

    Selection runs the shared rules of `common.base_reject` (the T0 reading at least 0.9 m less one cell, loops
    = boulders and loops_unsized = n + boulders) and the plan §7 family checks except the planning-range check,
    which only stops the build for review; the exact gap check (≥ 1.0 m) is the authoritative passage rule. The
    build runs everything else on the accepted candidate: `gz sdf -k`, the free-area tolerance, heights,
    determinism (T0 checks 1, 5, 6 and 8) and the planning-range check.
    """
    return base_reject(c.world, a) or first_failure(
        _family_checks(j, c.layout, c.gaps, c.world, a, ranges=False)
    )


@functools.cache
def instance(j: int) -> Candidate:
    """The first accepted candidate of setting j (seed SEED_BASE + 1000·j + i)."""
    return select(
        world_id(j),
        SEED_BASE + 1000 * j,
        functools.partial(_candidate, j),
        lambda c: _reject_reason(j, c, analyse(c.world)),
    )


def _family_checks(
    j: int, lay: Layout, gaps: Gaps, world: World, a: Analysis, *, ranges: bool
) -> list[CheckResult]:
    """The family checks of plan §7 on one candidate; `ranges` adds the planning-range check."""
    side, _ = SETTINGS[j]
    results: list[CheckResult] = []
    tol = 1e-9
    results.append(
        CheckResult(
            "F4 gaps ≥ 1.0 m (authoritative passage rule): exact smallest surface gap, trunk pairs, "
            "trunk–boulder disc, trunk–wall, boulder discs, disc–wall",
            min(
                gaps.surface,
                gaps.trunk_trunk,
                gaps.trunk_disc,
                gaps.trunk_wall,
                gaps.disc_disc,
                gaps.disc_wall,
            )
            >= GAP - tol,
            {
                "surface": round(gaps.surface, 4),
                "trunk_trunk": round(gaps.trunk_trunk, 4),
                "trunk_disc": round(gaps.trunk_disc, 4),
                "trunk_wall": round(gaps.trunk_wall, 4),
                "disc_disc": round(gaps.disc_disc, 4),
                "disc_wall": round(gaps.disc_wall, 4),
            },
        )
    )
    trunks = [p for p in world.primitives if p.kind is Kind.CYLINDER]
    boulders = [p for p in world.primitives if p.role is Role.LANDMARK]
    walls = [p for p in world.primitives if p.role is Role.BOUNDARY_WALL]
    radii = [p.size[0] for p in trunks]
    results.append(
        CheckResult(
            "F4 radii in [0.15, 0.40] m; heights 4.0 m trunks, 1.5 m boulders, 3.0 m walls",
            all(R_MIN <= r <= R_MAX for r in radii)
            and all(abs(p.height - TRUNK_H) < tol and p.z_min == 0.0 for p in trunks)
            and all(abs(p.height - BOULDER_H) < tol and p.z_min == 0.0 for p in boulders)
            and all(abs(p.height - WALL_H) < tol for p in walls),
            {
                "radius_min": min(radii, default=None),
                "radius_max": max(radii, default=None),
                "trunks": len(trunks),
                "boulders": len(boulders),
                "walls": len(walls),
            },
        )
    )
    results.append(
        CheckResult(
            "F4 trunk count equals the Poisson draw",
            len(trunks) == lay.poisson_n,
            {
                "n": len(trunks),
                "draw": lay.poisson_n,
                "density": round(len(trunks) / side**2, 6),
            },
        )
    )
    across = [min(b.long, b.short) for b in lay.boulders]
    longest = [max(b.long, b.short) for b in lay.boulders]
    rec_gaps = [
        abs(p - q)
        for centres in lay.recesses.values()
        for i, p in enumerate(centres)
        for q in centres[i + 1 :]
    ]
    results.append(
        CheckResult(
            "F4 landmarks: (S/20)² boulders ≥ 2 m across, S/10 recesses per wall ≥ 3.0 m apart",
            len(boulders) == (side // 20) ** 2
            and min(longest) >= 2.0 - tol
            and all(len(v) == side // 10 for v in lay.recesses.values())
            and min(rec_gaps) >= RECESS_GAP - tol,
            {
                "boulders": len(boulders),
                "longest_side_min_m": min(longest),
                "short_side_min_m": min(across),
                "recesses_per_wall": {w: len(v) for w, v in lay.recesses.items()},
                "min_recess_gap_m": round(min(rec_gaps), 2),
            },
        )
    )
    results.append(
        CheckResult(
            "F4 holes: loops_unsized = n + boulders, loops (≥ 2 m) = boulders",
            a.loops_unsized == len(trunks) + len(boulders) and a.loops == len(boulders),
            {"loops": a.loops, "loops_unsized": a.loops_unsized},
        )
    )
    results.append(walls_check(FAMILY, world, (Role.BOUNDARY_WALL,)))
    if ranges:
        lo, hi = RANGES[j]
        area = float(world.plan_data["construction_free_area_m2"])
        results.append(
            CheckResult(
                "F4 construction free area inside the plan's planning range (outside: stop for review)",
                lo <= area <= hi,
                area,
                f"range {RANGES[j]}; raster {a.free_area_m2:.2f} m²",
            )
        )
    return results


def _checks(j: int, lay: Layout, gaps: Gaps) -> ExtraCheck:
    """The build's family checks, bound to this candidate's layout and gaps."""

    def run(world: World, a: Analysis) -> list[CheckResult]:
        return _family_checks(j, lay, gaps, world, a, ranges=True)

    return run


def build(j: int) -> World:
    return instance(j).world


def builders() -> dict[str, Callable[[], World]]:
    return {world_id(j): functools.partial(build, j) for j in BUILT}
