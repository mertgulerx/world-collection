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

"""E3 Warehouse with moving objects (plan docs/plans/worlds/E3-warehouse-moving.md): R8 plus solid moving
objects at the high density (`E3-high`), drawn by the E0 engine (`worldgen/moving.py`). E3-low was removed by the
maintainer; the nested low density of E0 §6 stays the high draw's first n_low objects.

Every path is a piece of one of R8's reserved patrol paths (R8 §3.7, `PATROL_PATHS`), cut as in E3 §3.1; the
pieces are verified against R8's lines (`verify_lines`) and the aisle bands against R8's racks
(`verify_regions`) before anything is drawn. The world-specific checks E3-1 … E3-6 (E3 §7) run as extra checks.
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .. import checks, moving
from ..checks import Analysis
from ..moving import (
    DENSITY_AREA_M2,
    OBJECT_TYPES,
    CheckContext,
    DrawResult,
    DrawSpec,
    Limits,
    Mode,
    MovingDefinition,
    MovingObject,
    ObjectType,
    Template,
    Track,
    VelocityMode,
    Waypoint,
    blockage_state,
    check_context,
    doorway_regions,
    draw,
    footprint_distance,
    free_cells,
    longest_run,
    object_count,
    region_track,
    track,
    variant,
)
from ..named import r8_industrial_warehouse as r8
from ..raster import EIGHT
from ..world import BoolArray, CheckResult, FloatArray, Kind, Point, World

PLAN = "docs/plans/worlds/E3-warehouse-moving.md"
BASE_ID = "RP2"
SEED = 5301
# Fixed for the study (maintainer decision 36, E0 §3.3).
VELOCITY_MODE = VelocityMode.POSE_AND_VELOCITY
LIMITS = Limits()  # blockage limit X = 10 s, doorway dwell limit 12 s (E3 §7): the E0 defaults
DWELL_S = 3.0  # picking stop at each end of a run (E3 §3.1)
PASSING_GAP_M = 1.0  # E1 §3.1 passing-gap rule (E3-1)
MOUTH_LENGTH_M = 2.0  # aisle mouths (E3 §3)
CONVOY_MARGIN_M = 2.0  # convoy spacing rule (E0 §7, E3-6)
GAP_RAY_M = 10.0  # E3-1: free width is measured up to this distance from the run line
GAP_STEP_M = 0.01
TURN_STEP_RAD = 0.02  # E3-4: yaw samples of a turn at a loop corner
# Family "MO2" puts the world in the folder MO2_<slug> (T0 §3), as MO1 does.
FAMILY = "MO2"
WORLD_ID = "MO2"
SLUG = "warehouse_moving"
TITLE = "Warehouse with moving objects, high density"

# Type weights (E3 §3.2), renormalised by the engine over each template's allowed types.
TYPE_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("pallet_truck", 0.35),
    ("box", 0.30),
    ("cart", 0.20),
    ("cyl_l", 0.10),
    ("cyl_s", 0.05),
)
# Draw classes (E3 §3.2, maintainer 18 October 2026): object k takes class k mod 4, then the next classes in
# the cycle. Kept for E3-high reproducibility (the order fixes the draw); low removed.
CLASSES = ("staging_convoy", "rack_run", "east_convoy", "cross_north_run")

# Convoy slots carry heading-mode types only (maintainer 18 October 2026): every member then has the same corner
# turns and cycle time T, which E0 §7 requires (a slot whose T differs from its convoy's is discarded).
CONVOY_TYPES = ("pallet_truck", "cart", "box")
RACK_TYPES = ("pallet_truck", "cart", "box", "cyl_s", "cyl_l")
CROSS_TYPES = ("pallet_truck", "cart", "box")
NORTH_TYPES = ("pallet_truck", "cart", "box", "cyl_l")
RACK_AISLES = (("A1", -4.3), ("A2", 0.9), ("A4", 10.8))  # aisle, centreline y (R8 P-B)
RACK_PIECES = (("w", -23.0, -12.0), ("e", -2.0, 9.0))  # 3.0 m clear of aisle ends and cross aisle


@dataclass(frozen=True, slots=True)
class Line:
    """One E3 path template on an R8 reserved path: a loop (corners in travel order) or a patrol line."""

    template: str
    path: str  # R8 path id (R8 §3.7)
    mode: Mode
    points: tuple[Point, ...]
    draw_class: str
    allowed: tuple[str, ...]
    convoy_slots: int = 0

    def waypoints(self) -> tuple[Waypoint, ...]:
        dwell = DWELL_S if self.mode is Mode.PATROL else 0.0
        return tuple(Waypoint(x, y, dwell) for x, y in self.points)


LINES: tuple[Line, ...] = (
    Line(
        "staging_convoy",
        "P-A",
        Mode.LOOP,
        ((-24.0, -15.5), (10.0, -15.5), (10.0, -10.5), (-24.0, -10.5)),  # counter-clockwise
        "staging_convoy",
        CONVOY_TYPES,
        14,
    ),
    Line(
        "east_convoy",
        "P-E",
        Mode.LOOP,
        ((14.0, -6.0), (14.0, 7.0), (24.0, 7.0), (24.0, -6.0)),  # clockwise
        "east_convoy",
        CONVOY_TYPES,
        8,
    ),
    *(
        Line(
            f"rack_run_{aisle}_{side}",
            "P-B",
            Mode.PATROL,
            ((x0, y), (x1, y)),
            "rack_run",
            RACK_TYPES,
        )
        for aisle, y in RACK_AISLES
        for side, x0, x1 in RACK_PIECES
    ),
    Line(
        "cross_run",
        "P-C",
        Mode.PATROL,
        ((-7.0, -8.0), (-7.0, 13.5)),
        "cross_north_run",
        CROSS_TYPES,
    ),
    Line(
        "north_run_w",
        "P-D",
        Mode.PATROL,
        ((-26.0, 14.0), (-10.0, 14.0)),
        "cross_north_run",
        NORTH_TYPES,
    ),
    Line(
        "north_run_e",
        "P-D",
        Mode.PATROL,
        ((-4.0, 14.0), (12.5, 14.0)),
        "cross_north_run",
        NORTH_TYPES,
    ),
)
LINE_BY_TEMPLATE = {line.template: line for line in LINES}

# Aisle mouths (E3 §3): the 2.0 m end sections of A1, A2 and A4 at the west aisle (x = -26), both sides of the
# cross aisle (x = -9, -5) and the east zone (x = 12). id -> (x0, x1, y0, y1).
AISLE_Y = {"A1": (-5.8, -2.8), "A2": (-0.6, 2.4), "A4": (9.3, 12.3)}
MOUTH_X = (
    ("west", -26.0, 1.0),
    ("cross_w", -9.0, -1.0),
    ("cross_e", -5.0, 1.0),
    ("east", 12.0, -1.0),
)
AISLE_MOUTHS: dict[str, tuple[float, float, float, float]] = {
    f"{aisle}_{end}": (
        min(x, x + sign * MOUTH_LENGTH_M),
        max(x, x + sign * MOUTH_LENGTH_M),
        y0,
        y1,
    )
    for aisle, (y0, y1) in AISLE_Y.items()
    for end, x, sign in MOUTH_X
}
# E3-2 regions: every rack aisle (both halves, R8 §3.3) and both office rooms (R8 §3.5).
REGION_AISLE_Y = {**AISLE_Y, "A3": (4.6, 7.1)}
AISLE_HALVES = (("w", -26.0, -9.0), ("e", -5.0, 12.0))
RACK_AISLE_REGIONS: dict[str, tuple[float, float, float, float]] = {
    f"{aisle}_{half}": (x0, x1, y0, y1)
    for aisle, (y0, y1) in REGION_AISLE_Y.items()
    for half, x0, x1 in AISLE_HALVES
}
OFFICE_REGIONS = {f"office_{rid}": (*xs, *ys) for rid, xs, ys, _ in r8.OFFICE_ROOMS}


# ---------------------------------------------------------------------------------------------------------
# Templates and the draw


def _on_segment(p: Point, a: Point, b: Point) -> bool:
    """`p` lies on the segment a-b (exactly, for the axis-aligned R8 lines)."""
    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
    inside = min(a[0], b[0]) <= p[0] <= max(a[0], b[0]) and min(a[1], b[1]) <= p[1] <= max(
        a[1], b[1]
    )
    return abs(cross) < 1e-9 and inside


def verify_lines() -> None:
    """Every E3 line is a piece of its R8 reserved path; a loop uses all corners of its R8 rectangle."""
    paths = {pid: polylines for pid, _, polylines in r8.PATROL_PATHS}
    for line in LINES:
        polylines = paths[line.path]
        if line.mode is Mode.LOOP:
            corners = {p for poly in polylines for p in poly}
            if set(line.points) != corners:
                raise ValueError(f"MO2 {line.template}: corners differ from RP2 {line.path}")
            continue
        segments = [(a, b) for poly in polylines for a, b in zip(poly[:-1], poly[1:], strict=True)]
        if not any(all(_on_segment(p, a, b) for p in line.points) for a, b in segments):
            raise ValueError(f"MO2 {line.template}: not a piece of RP2 {line.path}")


def verify_regions() -> None:
    """The aisle bands and halves hard-coded above (E3-2, E3-3) are R8's rack aisles (read-only): each band
    lies between the facing rack rows, A1-A3 match R8's measured aisles, the halves match the rack ends, and
    the run centrelines are the band centres. Raises ValueError when R8 has changed."""
    racks = {name: (x0, x1, y0, y1) for name, x0, x1, y0, y1 in r8.RACKS}
    aisles = {name: (lo, hi) for name, _, _, lo, hi, _ in r8.AISLES}
    problems = []
    for k, (aisle, (y0, y1)) in enumerate(sorted(REGION_AISLE_Y.items()), start=1):
        for half, x0, x1 in AISLE_HALVES:
            south = racks[f"rack_r{k}_{half}"]
            if south[3] != y0:
                problems.append(f"{aisle}: south face {south[3]} != {y0}")
            if half == "e" and (south[0], south[1]) != (x0, x1):
                problems.append(f"{aisle}_{half}: rack x {south[:2]} != {(x0, x1)}")
            if half == "w" and south[1] != x1:
                problems.append(f"{aisle}_{half}: rack east end {south[1]} != {x1}")
        if aisle in aisles and aisles[aisle] != (y0, y1):
            problems.append(f"{aisle}: RP2 aisle {aisles[aisle]} != band {(y0, y1)}")
    if not math.isclose(AISLE_Y["A4"][1] - AISLE_Y["A4"][0], AISLE_Y["A1"][1] - AISLE_Y["A1"][0]):
        problems.append("A4 band is not as wide as A1")
    for aisle, y in RACK_AISLES:
        if not math.isclose(y, sum(AISLE_Y[aisle]) / 2):
            problems.append(f"{aisle}: run centreline {y} is not the band centre")
    west = {x0 for name, x0, *_ in r8.RACKS if name.endswith("_w") and name != "rack_r3_w"}
    ends = (AISLE_HALVES[0][1], AISLE_HALVES[0][2], AISLE_HALVES[1][1], AISLE_HALVES[1][2])
    if west != {ends[0]}:
        problems.append(f"west rack ends {west} != west half x {ends[0]}")
    if tuple(x for _, x, _ in MOUTH_X) != ends:
        problems.append(f"aisle mouths x {[x for _, x, _ in MOUTH_X]} != rack ends {ends}")
    if problems:
        raise ValueError("MO2: RP2 geometry differs from MO2's regions: " + "; ".join(problems))


def _instantiate(line: Line) -> moving.Instantiate:
    """E3 paths do not depend on the type: every line keeps at least 1.5 m from static geometry, more than
    r + 0.1 m for every allowed type (E3 §3.1)."""
    waypoints = line.waypoints()

    def make(_: ObjectType) -> Sequence[Waypoint]:
        return waypoints

    return make


def _class_order(k: int) -> Sequence[str]:
    i = k % len(CLASSES)
    return CLASSES[i:] + CLASSES[:i]


TEMPLATES: tuple[Template, ...] = tuple(
    Template(
        line.template,
        line.mode,
        line.allowed,
        _instantiate(line),
        draw_class=line.draw_class,
        convoy_slots=line.convoy_slots,
    )
    for line in LINES
)
SPEC = DrawSpec(SEED, TEMPLATES, TYPE_WEIGHTS, class_order=_class_order)


@functools.cache
def _base() -> tuple[World, Analysis, CheckContext]:
    """R8, its T0 analysis and the E0 check context (R8's doors as doorway regions)."""
    verify_lines()
    verify_regions()
    base = r8.build()
    base.validate()
    analysis = checks.analyse(base)
    return base, analysis, check_context(base, analysis, doorway_regions(base.doors), LIMITS)


@functools.cache
def draw_high() -> DrawResult:
    """The high-density draw (E0 §7)."""
    _, analysis, ctx = _base()
    return draw(SPEC, ctx, object_count(analysis.free_area_m2, DENSITY_AREA_M2["high"]))


def definition() -> MovingDefinition:
    """The high-density definition."""
    _, analysis, ctx = _base()
    result = draw_high()
    return MovingDefinition(
        "high",
        SEED,
        result.objects,
        VELOCITY_MODE,
        area_per_object_m2=DENSITY_AREA_M2["high"],
        free_area_m2=analysis.free_area_m2,
        doorways=ctx.doorways,
        base_world=BASE_ID,
        rejected_draws=len(result.rejections),
        limits=LIMITS,
    )


def build_high() -> World:
    return variant(
        _base()[0],
        world_id=WORLD_ID,
        slug=SLUG,
        title=TITLE,
        plan=PLAN,
        moving=definition(),
        family=FAMILY,
        extra_checks=(e3_checks,),
    )


# ---------------------------------------------------------------------------------------------------------
# World-specific checks E3-1 … E3-6 (E3 §7)


def _free_ray(a: Analysis, px: float, py: float, nx: float, ny: float) -> float:
    """Distance (m, to GAP_STEP_M) from (px, py) along (nx, ny) to the first occupied or off-grid cell."""
    g = a.grid
    s = np.arange(0.0, GAP_RAY_M + GAP_STEP_M / 2, GAP_STEP_M)
    cols = np.floor((px + s * nx - g.x0) / g.h).astype(np.int64)
    rows = np.floor((py + s * ny - g.y0) / g.h).astype(np.int64)
    inside = (rows >= 0) & (rows < g.ny) & (cols >= 0) & (cols < g.nx)
    blocked = ~inside
    blocked[inside] = a.occupied[rows[inside], cols[inside]]
    hit = np.flatnonzero(blocked)
    return float(s[hit[0]]) if len(hit) else GAP_RAY_M


def passing_gap(defn: MovingDefinition, a: Analysis) -> CheckResult:
    """MO2-1: at every run point, the free width beside the object (to static geometry, on the wider side) is
    at least 1.0 m. Runs are the patrol templates; convoys move through open floor (E3 §3.1)."""
    worst: dict[str, float] = {}
    narrow_side: dict[str, float] = {}
    for obj in defn.objects:
        if obj.mode is not Mode.PATROL:
            continue
        # half-width across the run: a cylinder's radius, half the body width of a heading-mode box
        half = obj.object_type.half_extent[1]
        w0, w1 = obj.waypoints[0], obj.waypoints[-1]
        length = math.sqrt((w1.x - w0.x) ** 2 + (w1.y - w0.y) ** 2)
        ux, uy = (w1.x - w0.x) / length, (w1.y - w0.y) / length
        wide, narrow = math.inf, math.inf
        for s in np.linspace(0.0, length, math.ceil(length / 0.05) + 1):
            px, py = w0.x + s * ux, w0.y + s * uy
            left = _free_ray(a, px, py, -uy, ux) - half
            right = _free_ray(a, px, py, uy, -ux) - half
            wide, narrow = min(wide, max(left, right)), min(narrow, min(left, right))
        worst[obj.id] = round(wide, 3)
        narrow_side[obj.id] = round(narrow, 3)
    value = min(worst.values()) if worst else None
    return CheckResult(
        "MO2-1 passing gap",
        value is None or value >= PASSING_GAP_M - 1e-9,
        value,
        f">= {PASSING_GAP_M} m on the wider side at every run point; per run (wider, narrower side): "
        + ", ".join(f"{k} {worst[k]}/{narrow_side[k]}" for k in worst),
    )


def _region_mask(a: Analysis, box: tuple[float, float, float, float]) -> BoolArray:
    g = a.grid
    xs, ys = g.centres()
    x0, x1, y0, y1 = box
    mask: BoolArray = ((ys >= y0) & (ys <= y1))[:, None] & ((xs >= x0) & (xs <= x1))[None, :]
    return mask


def regions_cut_off(defn: MovingDefinition, a: Analysis, ctx: CheckContext) -> CheckResult:
    """MO2-2: M6 restricted to the cells of each office room and rack aisle. An instant counts for a region
    when the cells of B(t) inside it cover at least 1.0 m² (M6's room threshold). Only instants where U(t) has
    more than one component can count; those are labelled in full."""
    state = blockage_state(defn.objects, ctx)
    regions = {
        name: _region_mask(a, box) & ctx.free_eroded
        for name, box in {**RACK_AISLE_REGIONS, **OFFICE_REGIONS}.items()
    }
    g = ctx.grid
    need = math.ceil(ctx.limits.blocked_area_m2 / (g.h * g.h) - 1e-9)
    candidates = np.flatnonzero(state.split)
    flags = {name: np.zeros(len(ctx.times_ns), dtype=bool) for name in regions}
    for i in candidates:
        u = free_cells(state.tracks, int(i), ctx)
        labels, n = ndimage.label(u, structure=EIGHT)
        if n <= 1:
            continue
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        cut = (labels != int(np.argmax(sizes))) & u
        for name, mask in regions.items():
            if np.count_nonzero(cut & mask) >= need:
                flags[name][i] = True
    runs = {name: round(longest_run(f) * ctx.limits.sample_dt_s, 1) for name, f in flags.items()}
    worst = max(runs.values())
    return CheckResult(
        "MO2-2 regions",
        worst <= ctx.limits.blockage_limit_s + 1e-9,
        worst,
        f"<= {ctx.limits.blockage_limit_s} s per office room and rack aisle; {len(candidates)} instants with "
        f"U(t) in more than one component; regions with a cut-off run: "
        + (", ".join(f"{k} {v} s" for k, v in runs.items() if v > 0) or "none"),
    )


def mouth_dwell(defn: MovingDefinition, tracks: Sequence[Track], ctx: CheckContext) -> CheckResult:
    """MO2-3: each interval in which an object's footprint, inflated by 0.29 m, intersects an aisle mouth lasts
    at most the doorway dwell limit (12.0 s)."""
    lim = ctx.limits
    n = len(ctx.times_ns)
    worst: dict[str, float] = {}
    for obj, tr in zip(defn.objects, tracks, strict=True):
        for name, (x0, x1, y0, y1) in AISLE_MOUTHS.items():
            cx, cy, hx, hy = (x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0) / 2, (y1 - y0) / 2
            # circumscribed-disc prefilter: skip a mouth the object never comes near
            gx = np.maximum(np.abs(tr.x - cx) - hx, 0.0)
            gy = np.maximum(np.abs(tr.y - cy) - hy, 0.0)
            if float(np.sqrt(gx * gx + gy * gy).min()) > tr.radius + lim.robot_radius_m:
                continue
            inside = footprint_distance(tr, region_track(cx, cy, hx, hy, n)) <= lim.robot_radius_m
            run = longest_run(inside) * lim.sample_dt_s
            if run > 0:
                key = f"{obj.id} {name}"
                worst[key] = round(max(worst.get(key, 0.0), run), 1)
    value = max(worst.values()) if worst else 0.0
    return CheckResult(
        "MO2-3 aisle mouth dwell",
        value <= lim.doorway_dwell_limit_s + 1e-9,
        value,
        f"<= {lim.doorway_dwell_limit_s} s; {len(AISLE_MOUTHS)} mouths; longest per object and mouth: "
        + (", ".join(f"{k} {v} s" for k, v in sorted(worst.items())) or "none"),
    )


def _rect_boundary(
    x: float, y: float, yaw: float, hx: float, hy: float
) -> tuple[FloatArray, FloatArray]:
    """Points every 0.05 m (at most) on a rotated rectangle's outline."""
    us, vs = [], []
    nu, nv = max(2, math.ceil(2 * hx / 0.05) + 1), max(2, math.ceil(2 * hy / 0.05) + 1)
    for u in np.linspace(-hx, hx, nu):
        us += [u, u]
        vs += [-hy, hy]
    for v in np.linspace(-hy, hy, nv):
        us += [-hx, hx]
        vs += [v, v]
    u_arr, v_arr = np.array(us), np.array(vs)
    c, s = math.cos(yaw), math.sin(yaw)
    return x + u_arr * c - v_arr * s, y + u_arr * s + v_arr * c


def rack_faces(defn: MovingDefinition, a: Analysis, ctx: CheckContext) -> CheckResult:
    """MO2-4: M1 for the rotated rectangles of carts, pallet trucks and boxes at their heading, along every
    segment and through every corner turn: min over outline points of the distance-transform clearance,
    minus 0.1 m."""
    g = a.grid
    margin = ctx.limits.static_margin_m
    worst: dict[str, float] = {}
    for obj in defn.objects:
        t = obj.object_type
        if t.kind is not Kind.BOX:
            continue
        hx, hy = t.half_extent
        poses: list[tuple[float, float, float]] = []
        for ph in obj.timeline.phases:
            if ph.kind is moving.PhaseKind.MOVE:
                length = math.sqrt(ph.dx * ph.dx + ph.dy * ph.dy)
                for f in np.linspace(0.0, 1.0, math.ceil(length / 0.05) + 1):
                    poses.append((ph.x0 + f * ph.dx, ph.y0 + f * ph.dy, ph.yaw0))
            elif ph.kind is moving.PhaseKind.TURN:
                for f in np.linspace(0.0, 1.0, math.ceil(abs(ph.dyaw) / TURN_STEP_RAD) + 1):
                    poses.append((ph.x0, ph.y0, ph.yaw0 + f * ph.dyaw))
        best = math.inf
        for x, y, yaw in poses:
            px, py = _rect_boundary(x, y, yaw, hx, hy)
            cols = np.floor((px - g.x0) / g.h).astype(np.int64)
            rows = np.floor((py - g.y0) / g.h).astype(np.int64)
            inside = (rows >= 0) & (rows < g.ny) & (cols >= 0) & (cols < g.nx)
            clear = np.zeros(len(px))  # off-grid points count as clearance 0, as in M1
            clear[inside] = a.clearance[rows[inside], cols[inside]]
            best = min(best, float(clear.min()))
        worst[obj.id] = round(best - margin, 3)
    value = min(worst.values()) if worst else None
    return CheckResult(
        "MO2-4 rack faces",
        value is None or value >= -1e-9,
        value,
        "min over rectangle outlines (heading, corner turns) of clearance - 0.1 m; per object: "
        + ", ".join(f"{k} {v}" for k, v in worst.items()),
    )


def nesting(world: World) -> CheckResult:
    """MO2-5: the world's objects equal the first n of the high density. Trivially true for E3-high (the only
    E3 world since E3-low was removed); kept so the check set of E3 §7 stays complete."""
    defn = world.moving
    assert defn is not None
    high = definition()
    same = defn.objects == high.objects[: defn.count]
    return CheckResult(
        "MO2-5 nesting",
        same,
        f"{defn.count} of {high.count}",
        f"{defn.density} density objects equal the first {defn.count} high-density objects",
    )


def templates(defn: MovingDefinition, tracks: Sequence[Track], ctx: CheckContext) -> CheckResult:
    """MO2-6: footprints on different templates never overlap at a sampled time (minimum distance reported),
    and each convoy's spacing (loop length / slots) exceeds its largest allowed footprint length + 2.0 m."""
    best = math.inf
    pair = ""
    objs = defn.objects
    for i in range(len(objs)):
        for j in range(i + 1, len(objs)):
            if objs[i].template == objs[j].template:
                continue
            a, b = tracks[i], tracks[j]
            dx, dy = a.x - b.x, a.y - b.y
            centre = np.sqrt(dx * dx + dy * dy) - a.radius - b.radius
            # circumscribed discs give a lower bound, exact where they are apart; exact where near
            low = float(centre.min())
            near = np.flatnonzero(centre < 0.5)
            if len(near):
                low = max(low, float(footprint_distance(a.at(near), b.at(near)).min()))
            if low < best:
                best, pair = low, f"{objs[i].id}/{objs[j].id}"
    spacing = {}
    ok_spacing = True
    for line in LINES:
        if not line.convoy_slots:
            continue
        loop = moving.path_length(line.waypoints(), Mode.LOOP) / line.convoy_slots
        need = max(OBJECT_TYPES[t].footprint_length for t in line.allowed) + CONVOY_MARGIN_M
        spacing[line.template] = f"{loop:.2f} > {need:.2f} m"
        ok_spacing &= loop > need
    value = None if math.isinf(best) else round(best, 3)
    return CheckResult(
        "MO2-6 templates",
        (value is None or value > 0.0) and ok_spacing,
        value,
        f"min footprint distance across templates ({pair or 'no pair'}); convoy spacing: "
        + ", ".join(f"{k} {v}" for k, v in spacing.items()),
    )


def e3_checks(world: World, a: Analysis) -> list[CheckResult]:
    defn = world.moving
    assert defn is not None
    ctx = check_context(world, a, defn.doorways, defn.limits)
    tracks = [track(obj, ctx.times_ns) for obj in defn.objects]
    return [
        passing_gap(defn, a),
        regions_cut_off(defn, a, ctx),
        mouth_dwell(defn, tracks, ctx),
        rack_faces(defn, a, ctx),
        nesting(world),
        templates(defn, tracks, ctx),
    ]


__all__ = ["MovingObject", "build_high", "definition", "draw_high", "e3_checks"]
