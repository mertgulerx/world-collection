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

"""E1 Office strip with moving objects.

The B4 office strip with solid moving objects at the high density (`E1-high`; E1-low was removed by the
maintainer, the nested low density of E0 §6 stays its first n_low objects). The geometry below is checked against
B4's constants before anything is drawn (`verify_base`). The path templates of E1 §3.1, in candidate order:

- `corridor_lane_a` (south side) and `corridor_lane_b` (north side): patrols along the corridor, centreline
  r + 0.15 m from the side wall face, ends r + 0.15 m before the end walls, dwell 2.0 s at each end. A lane is
  split at every spawn whose 1.0 m clearance (M3) it would break; each piece is its own candidate when at least
  one allowed type gives it 3.0 m or more (pieces west to east). A piece end within r + 0.29 m of a
  doorway region moves inward to the first point clear of it (M5 forbids dwells there). Types failing the
  passing-gap rule are not allowed on the lane.
- `room_visit_k` (k = 1 ... 10, B4 room order R01 ... R10): a `cyl_s` patrol from the lane point opposite door k
  through the door centre to the room centre (dwell 0, 0, 5.0 s). Rooms that contain a spawn (R09, S3) get no
  room template.
- `room_loop_k`: a loop round the square inset 1.0 m from room k's inner wall faces, corners counter-clockwise
  from the south-west corner, dwell 0.

Draw classes (E1 §3.2): object k takes a corridor candidate when k is even and a room candidate when k is odd,
falling back to the other class when its own has none left.
"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
from scipy import ndimage

from .. import checks
from ..checks import Analysis
from ..diagnostic import b4_office_strip as b4
from ..moving import (
    DENSITY_AREA_M2,
    OBJECT_TYPES,
    DoorwayRegion,
    DrawResult,
    DrawSpec,
    Limits,
    Mode,
    MovingDefinition,
    MovingObject,
    ObjectType,
    Template,
    VelocityMode,
    Waypoint,
    blockage_state,
    check_context,
    doorway_regions,
    draw,
    free_cells,
    longest_run,
    object_count,
    pass_pairs,
    path_length,
    variant,
)
from ..raster import EIGHT
from ..world import CheckResult, Kind, World

SEED = 5101
BASE_WORLD = "SC4"
# Fixed for the study (maintainer decision 36, 18 October 2026).
VELOCITY_MODE = VelocityMode.POSE_AND_VELOCITY
TYPE_WEIGHTS = (("cyl_s", 0.5), ("box", 0.3), ("cart", 0.2))
LANE_TYPES = ("cyl_s", "box", "cart")
VISIT_TYPES = ("cyl_s",)
LOOP_TYPES = ("cyl_s", "box")
CORRIDOR_CLASS = "corridor"
ROOM_CLASS = "room"

# B4 geometry (plan B4 §3, `b4_office_strip.build`): corridor inner faces and the outer end walls
CORRIDOR_X = (-12.9, 12.9)
CORRIDOR_Y = (-1.5, 1.5)
CORRIDOR_WIDTH_M = CORRIDOR_Y[1] - CORRIDOR_Y[0]
DOOR_WALL_Y = {"s": -1.6, "n": 1.6}  # corridor wall centrelines carrying the doors

LANE_MARGIN_M = (
    0.15  # lane centreline r + 0.15 m from the wall face; lane ends r + 0.15 m before end walls
)
LANE_DWELL_S = 2.0
MIN_LANE_M = 3.0
PASSING_GAP_M = 1.0
VISIT_DWELL_S = 5.0
LOOP_INSET_M = 1.0
ROOM_BLOCKAGE_LIMIT_S = 10.0
ROBOT_RADIUS_M = Limits().robot_radius_m  # MOGI's circumscribed radius, the M5 inflation (E0 §8)
QUANTUM_M = 1e-4  # waypoints are written with 0.1 mm (E0 §6)


# ---------------------------------------------------------------------------------------------------------
# Geometry helpers


def _away(value: float, direction: float) -> float:
    """`value` rounded to 0.1 mm towards `direction` (+1 up, -1 down): no quantisation moves a waypoint
    closer to the wall or spawn it was placed against."""
    scaled = round(value / QUANTUM_M, 6)
    steps = math.ceil(scaled) if direction > 0 else math.floor(scaled)
    return round(steps * QUANTUM_M, 4) + 0.0


def _across(t: ObjectType) -> float:
    """r_side: the footprint's half-width across a lane (heading mode keeps body x along the lane)."""
    return t.half_extent[1]


def lane_y(side: str, t: ObjectType) -> float:
    """Lane centreline y for type t: r + 0.15 m from the south ('a') or north ('b') wall face."""
    offset = t.radius + LANE_MARGIN_M
    if side == "a":
        return _away(CORRIDOR_Y[0] + offset, +1)
    return _away(CORRIDOR_Y[1] - offset, -1)


def passing_gap(side: str, t: ObjectType) -> float:
    """Free width beside a lane object on the corridor's far side (E1 §3.1)."""
    y = lane_y(side, t)
    offset = y - CORRIDOR_Y[0] if side == "a" else CORRIDOR_Y[1] - y
    return CORRIDOR_WIDTH_M - offset - _across(t)


def _spawn_half_gap(t: ObjectType, dy: float, clearance: float) -> float | None:
    """Half-length of the lane interval, centred on a spawn's x, where the footprint comes closer than
    `clearance` to the spawn; None when the lane never does. The footprint keeps its lane heading (patrols
    reverse without turning)."""
    hx, hy = t.half_extent
    if t.kind is Kind.CYLINDER:
        reach = clearance + hx
        return math.sqrt(reach * reach - dy * dy) if abs(dy) < reach else None
    ey = abs(dy) - hy
    if ey >= clearance:
        return None
    if ey <= 0:
        return hx + clearance
    return hx + math.sqrt(clearance * clearance - ey * ey)


def _lane_ends(t: ObjectType) -> tuple[float, float]:
    return (
        _away(CORRIDOR_X[0] + t.radius + LANE_MARGIN_M, +1),
        _away(CORRIDOR_X[1] - t.radius - LANE_MARGIN_M, -1),
    )


def _spawn_cut(base: World, side: str, t: ObjectType, spawn: str) -> tuple[float, float] | None:
    """The lane interval a spawn removes for type t (M3, 1.0 m at all times), or None."""
    s = next(s for s in base.spawns if s.id == spawn)
    x_lo, x_hi = _lane_ends(t)
    half = _spawn_half_gap(t, lane_y(side, t) - s.y, 1.0)
    if half is None or s.x + half <= x_lo or s.x - half >= x_hi:
        return None
    return _away(s.x - half - QUANTUM_M, -1), _away(s.x + half + QUANTUM_M, +1)


def split_spawns(base: World, side: str, types: tuple[str, ...]) -> tuple[str, ...]:
    """Spawns that split the lane, west to east. Every allowed type must be split by the same spawns, so
    piece i means the same stretch of corridor for every type."""
    by_type = {
        n: tuple(s.id for s in base.spawns if _spawn_cut(base, side, OBJECT_TYPES[n], s.id))
        for n in types
    }
    if len(set(by_type.values())) > 1:
        raise ValueError(f"corridor_lane_{side}: spawn splits differ between types: {by_type}")
    ids = next(iter(by_type.values()), ())
    xs = {s.id: s.x for s in base.spawns}
    return tuple(sorted(ids, key=lambda i: xs[i]))


def lane_pieces(
    base: World, side: str, t: ObjectType, spawns: tuple[str, ...]
) -> list[tuple[float, float] | None]:
    """Piece i of the lane for type t: the stretch between split spawn i - 1 and split spawn i (the lane ends
    outside them); None when the stretch is empty."""
    x_lo, x_hi = _lane_ends(t)
    bounds = [x_lo]
    for sid in spawns:
        cut = _spawn_cut(base, side, t, sid)
        assert cut is not None
        bounds += [cut[0], cut[1]]
    bounds.append(x_hi)
    y = lane_y(side, t)
    reach = t.radius + ROBOT_RADIUS_M
    regions = doorway_regions(base.doors)
    pieces: list[tuple[float, float] | None] = []
    for i in range(0, len(bounds), 2):
        a, b = _clear_of_doorways(max(bounds[: i + 1]), bounds[i + 1], y, reach, regions)
        pieces.append((a, b) if b > a else None)
    return pieces


def _clear_of_doorways(
    a: float, b: float, y: float, reach: float, regions: tuple[DoorwayRegion, ...]
) -> tuple[float, float]:
    """Move the piece ends inward along the lane to the first points farther than `reach` (r + 0.29 m) from
    every doorway region, so their 2.0 s dwells pass M5 (maintainer, 18 October 2026)."""
    moved = True
    while moved and b > a:
        moved = False
        for r in regions:
            ey = max(abs(y - r.cy) - r.hy, 0.0)
            if ey >= reach:
                continue
            half = r.hx + math.sqrt(reach * reach - ey * ey)  # x interval of points within reach
            lo, hi = r.cx - half, r.cx + half
            if lo <= a <= hi:
                a, moved = _away(hi + QUANTUM_M, +1), True
            if lo <= b <= hi:
                b, moved = _away(lo - QUANTUM_M, -1), True
    return a, b


def _rooms() -> list[tuple[str, str, float, float, float, float, float]]:
    """(room id, side, x0, x1, y0, y1, door x) of B4's rooms in room order (doors D01 ... D10)."""
    out = []
    for rid, side, (x0, x1), cx in b4.ROOMS:
        y0, y1 = b4.room_y(side)
        out.append((rid, side, x0, x1, y0, y1, cx))
    return out


def verify_base(base: World) -> None:
    """The B4 geometry hard-coded above matches B4 as built (read-only): the corridor faces and end walls
    bound the rooms, and the door walls carry B4's doors. Raises ValueError when B4 has changed."""
    problems = []
    rooms = _rooms()
    for side in DOOR_WALL_Y:
        ends = (min(r[2] for r in rooms if r[1] == side), max(r[3] for r in rooms if r[1] == side))
        if ends != CORRIDOR_X:
            problems.append(f"side {side} rooms span x {ends}, corridor ends {CORRIDOR_X}")
    for rid, side, x0, x1, y0, y1, _ in rooms:
        face = y1 if side == "s" else y0  # the room's corridor-side face
        wall = DOOR_WALL_Y[side]
        corridor = CORRIDOR_Y[0] if side == "s" else CORRIDOR_Y[1]
        if not math.isclose(abs(face - corridor), 0.2) or not math.isclose(
            (face + corridor) / 2, wall
        ):
            problems.append(f"{rid}: corridor wall {face}..{corridor} is not centred on y {wall}")
    if len(base.doors) != len(rooms):
        problems.append(f"{len(base.doors)} doors for {len(rooms)} rooms")
    for d, (rid, side, *_, cx) in zip(base.doors, rooms, strict=False):
        if (d.x, d.y, d.wall_axis) != (cx, DOOR_WALL_Y[side], "x"):
            problems.append(f"door {d.id} at ({d.x}, {d.y}, {d.wall_axis}) is not room {rid}'s")
    if problems:
        raise ValueError("MO1: SC4 geometry differs from MO1's constants: " + "; ".join(problems))


# ---------------------------------------------------------------------------------------------------------
# Templates (E1 §3.1)


def _lane_template(
    base: World, side: str, index: int, types: tuple[str, ...], spawns: tuple[str, ...]
) -> Template:
    def instantiate(t: ObjectType) -> list[Waypoint] | None:
        piece = lane_pieces(base, side, t, spawns)[index]
        if piece is None:
            return None
        a, b = piece
        y = lane_y(side, t)
        return [Waypoint(a, y, LANE_DWELL_S), Waypoint(b, y, LANE_DWELL_S)]

    return Template(
        f"corridor_lane_{side}",
        Mode.PATROL,
        types,
        instantiate,
        CORRIDOR_CLASS,
        min_length_m=MIN_LANE_M,
    )


def _visit_template(
    k: int, side: str, x0: float, x1: float, y0: float, y1: float, cx: float
) -> Template:
    lane = "a" if side == "s" else "b"

    def instantiate(t: ObjectType) -> list[Waypoint]:
        return [
            Waypoint(cx, lane_y(lane, t), 0.0),
            Waypoint(cx, DOOR_WALL_Y[side], 0.0),
            Waypoint((x0 + x1) / 2, (y0 + y1) / 2, VISIT_DWELL_S),
        ]

    return Template(f"room_visit_{k}", Mode.PATROL, VISIT_TYPES, instantiate, ROOM_CLASS)


def _loop_template(k: int, x0: float, x1: float, y0: float, y1: float) -> Template:
    a0, a1, b0, b1 = x0 + LOOP_INSET_M, x1 - LOOP_INSET_M, y0 + LOOP_INSET_M, y1 - LOOP_INSET_M

    def instantiate(t: ObjectType) -> list[Waypoint]:
        return [Waypoint(a0, b0), Waypoint(a1, b0), Waypoint(a1, b1), Waypoint(a0, b1)]

    return Template(f"room_loop_{k}", Mode.LOOP, LOOP_TYPES, instantiate, ROOM_CLASS)


def lane_types(side: str) -> tuple[str, ...]:
    """The lane's allowed types: those satisfying the passing-gap rule."""
    return tuple(
        n for n in LANE_TYPES if passing_gap(side, OBJECT_TYPES[n]) >= PASSING_GAP_M - 1e-9
    )


def templates(base: World) -> tuple[Template, ...]:
    out: list[Template] = []
    for side in ("a", "b"):
        types = lane_types(side)
        spawns = split_spawns(base, side, types)
        for index in range(len(spawns) + 1):
            pieces = [lane_pieces(base, side, OBJECT_TYPES[n], spawns)[index] for n in types]
            if max((p[1] - p[0] for p in pieces if p is not None), default=0.0) >= MIN_LANE_M:
                out.append(_lane_template(base, side, index, types, spawns))
    # rooms that contain a spawn (R09, S3) get no room template (maintainer, 18 October 2026)
    rooms = [
        (k, room)
        for k, room in enumerate(_rooms(), start=1)
        if not any(room[2] < s.x < room[3] and room[4] < s.y < room[5] for s in base.spawns)
    ]
    for k, (_, side, x0, x1, y0, y1, cx) in rooms:
        out.append(_visit_template(k, side, x0, x1, y0, y1, cx))
    for k, (_, _, x0, x1, y0, y1, _) in rooms:
        out.append(_loop_template(k, x0, x1, y0, y1))
    return tuple(out)


def class_order(k: int) -> tuple[str, ...]:
    return (CORRIDOR_CLASS, ROOM_CLASS) if k % 2 == 0 else (ROOM_CLASS, CORRIDOR_CLASS)


# ---------------------------------------------------------------------------------------------------------
# Drawing and definitions


@cache
def _base() -> tuple[World, Analysis]:
    base = b4.build()
    base.validate()
    verify_base(base)
    return base, checks.analyse(base)


@cache
def draw_high() -> DrawResult:
    """The high-density draw (E0 §7)."""
    base, analysis = _base()
    ctx = check_context(base, analysis, doorway_regions(base.doors))
    spec = DrawSpec(SEED, templates(base), TYPE_WEIGHTS, class_order=class_order)
    return draw(spec, ctx, object_count(analysis.free_area_m2, DENSITY_AREA_M2["high"]))


@cache
def high_definition() -> MovingDefinition:
    base, analysis = _base()
    result = draw_high()
    return MovingDefinition(
        "high",
        SEED,
        result.objects,
        VELOCITY_MODE,
        area_per_object_m2=DENSITY_AREA_M2["high"],
        free_area_m2=analysis.free_area_m2,
        doorways=doorway_regions(base.doors),
        base_world=BASE_WORLD,
        rejected_draws=len(result.rejections),
    )


# ---------------------------------------------------------------------------------------------------------
# World-specific checks E1-1 ... E1-5 (E1 §7)


def _is_lane(obj: MovingObject) -> bool:
    return obj.template.startswith("corridor_lane_")


def _segment_hits_rect(
    ax: float, ay: float, bx: float, by: float, x0: float, x1: float, y0: float, y1: float
) -> bool:
    """Whether segment a-b meets the closed rectangle (Liang-Barsky clipping)."""
    t0, t1 = 0.0, 1.0
    dx, dy = bx - ax, by - ay
    for p, q in ((-dx, ax - x0), (dx, x1 - ax), (-dy, ay - y0), (dy, y1 - ay)):
        if p == 0:
            if q < 0:
                return False
            continue
        r = q / p
        if p < 0:
            t0 = max(t0, r)
        else:
            t1 = min(t1, r)
        if t0 > t1:
            return False
    return True


def _check_passing_gap(objects: tuple[MovingObject, ...]) -> CheckResult:
    worst: float | None = None
    bad = []
    for obj in objects:
        if not _is_lane(obj):
            continue
        t = obj.object_type
        y = obj.waypoints[0].y
        offset = min(y - CORRIDOR_Y[0], CORRIDOR_Y[1] - y)
        gap = CORRIDOR_WIDTH_M - offset - _across(t)
        if offset < t.radius + LANE_MARGIN_M - 1e-9 or gap < PASSING_GAP_M - 1e-9:
            bad.append(obj.id)
        worst = gap if worst is None else min(worst, gap)
    allowed = {side: list(lane_types(side)) for side in ("a", "b")}
    return CheckResult(
        "MO1-1 passing gap",
        not bad,
        None if worst is None else round(worst, 4),
        f"min free width beside a lane object >= {PASSING_GAP_M} m; lane types {allowed}; "
        f"failing: {', '.join(bad) or 'none'}",
    )


def _check_door_use(base: World, objects: tuple[MovingObject, ...]) -> CheckResult:
    use: dict[str, list[str]] = {}
    bad = []
    for d in base.doors:
        x0, x1 = d.x - d.width / 2, d.x + d.width / 2
        y0, y1 = d.y - 0.1, d.y + 0.1  # the gap: door width × 0.2 m wall
        for obj in objects:
            w = obj.waypoints
            if any(
                _segment_hits_rect(w[a].x, w[a].y, w[b].x, w[b].y, x0, x1, y0, y1)
                for a, b in pass_pairs(len(w), obj.mode)
            ):
                use.setdefault(d.id, []).append(obj.id)
                if obj.type != "cyl_s" or not obj.template.startswith("room_visit_"):
                    bad.append(f"{obj.id} ({obj.type}, {obj.template}) crosses {d.id}")
    for door_id, ids in use.items():
        if len(ids) > 1:
            bad.append(f"{door_id} used by {len(ids)} paths")
    return CheckResult(
        "MO1-2 door use",
        not bad,
        {k: v for k, v in sorted(use.items())},
        "; ".join(bad) or "each door at most one room_visit path; only cyl_s crosses doors",
    )


def _check_room_blockage(world: World, analysis: Analysis) -> CheckResult:
    """No room is cut off from the corridor for more than 10 s: none of its eroded free cells in R(t). Only
    the instants where M6 finds U(t) in more than one component can cut a room off; those are labelled."""
    defn = world.moving
    assert defn is not None
    ctx = check_context(world, analysis, defn.doorways, defn.limits)
    xs, ys = ctx.grid.centres()
    rooms = []
    for rid, _, x0, x1, y0, y1, _ in _rooms():
        cols = (xs > x0) & (xs < x1)
        rows = (ys > y0) & (ys < y1)
        mask = np.zeros_like(ctx.free_eroded)
        mask[np.ix_(rows, cols)] = True
        rooms.append((rid, mask & ctx.free_eroded))
    state = blockage_state(defn.objects, ctx)
    cut = np.zeros((len(rooms), len(ctx.times_ns)), dtype=bool)
    for i in (int(v) for v in np.flatnonzero(state.split)):
        labels, count = ndimage.label(free_cells(state.tracks, i, ctx), structure=EIGHT)
        if count <= 1:
            continue
        sizes = np.bincount(labels.ravel())
        sizes[0] = 0
        main = int(sizes.argmax())
        for j, (_, cells) in enumerate(rooms):
            cut[j, i] = not bool((labels[cells] == main).any())
    dt = ctx.limits.sample_dt_s
    longest = {rid: round(longest_run(cut[j]) * dt, 1) for j, (rid, _) in enumerate(rooms)}
    worst = max(longest.values())
    return CheckResult(
        "MO1-3 room blockage",
        worst <= ROOM_BLOCKAGE_LIMIT_S + 1e-9,
        worst,
        f"longest cut-off per room (s): {longest}; limit {ROOM_BLOCKAGE_LIMIT_S} s",
    )


def _check_lane_length(objects: tuple[MovingObject, ...]) -> CheckResult:
    lengths = {o.id: round(path_length(o.waypoints, o.mode), 4) for o in objects if _is_lane(o)}
    return CheckResult(
        "MO1-4 lane length",
        all(v >= MIN_LANE_M - 1e-9 for v in lengths.values()),
        lengths,
        f">= {MIN_LANE_M} m",
    )


def _check_nesting(objects: tuple[MovingObject, ...]) -> CheckResult:
    """MO1-5: the objects are the first n of the high density. Trivially true for E1-high (the only E1 world
    since E1-low was removed); kept so the check set of E1 §7 stays complete."""
    high = high_definition().objects
    same = objects == high[: len(objects)]
    return CheckResult(
        "MO1-5 nesting",
        same,
        {"objects": len(objects), "high": len(high)},
        "objects equal the first n of the high density, value for value",
    )


def _checks(world: World, analysis: Analysis) -> list[CheckResult]:
    defn = world.moving
    assert defn is not None
    base, _ = _base()
    return [
        _check_passing_gap(defn.objects),
        _check_door_use(base, defn.objects),
        _check_room_blockage(world, analysis),
        _check_lane_length(defn.objects),
        _check_nesting(defn.objects),
    ]


# ---------------------------------------------------------------------------------------------------------
# Worlds


def build_high() -> World:
    base, _ = _base()
    # family "MO1" puts the world in the folder MO1_office_strip_moving (T0 §3)
    return variant(
        base,
        world_id="MO1",
        slug="office_strip_moving",
        title="Office strip with moving objects (high density)",
        moving=high_definition(),
        family="MO1",
        extra_checks=(_checks,),
    )
