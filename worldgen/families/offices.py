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

"""F2 Offices, a generated family of eight office floors (plan docs/plans/worlds/C2-offices.md).

Lengths of the construction are integers in units of 0.1 m measured from the floor's south-west corner (the
inner faces of the outer wall), so every wall lies on the 0.1 m construction grid (plan §3.2).
"""

from __future__ import annotations

import functools
import math
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from ..checks import Analysis, analyse
from ..geometry import box, door, grid_walls, walled_rectangle
from ..layout_checks import door_openings, rect_area
from ..world import (
    BoolArray,
    CheckResult,
    Dimension,
    Door,
    Expected,
    ExtraCheck,
    FamilyInfo,
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
    snap_units,
    streams,
    walls_check,
)

FAMILY = "F2"
PLAN = "docs/plans/worlds/C2-offices.md"
SEED_BASE = 200000  # setting j tries seeds SEED_BASE + 1000·j + i (plan §3.3)
UNIT = 0.1
CORRIDOR_HALF = 10  # clear half-width, 1.0 m
WALL = 2  # wall thickness, 0.2 m
DOOR = 12  # door width, 1.2 m
DOOR_MARGIN = 11  # door centre to a segment end: 0.5 m margin + half the door
MIN_SIDE = 25  # minimum room clear side, 2.5 m
MIN_OVERLAP = 20  # rooms facing across a wall are adjacent when their extents overlap by 2.0 m
MIN_SEGMENT = 22  # a segment takes a door when it is at least 2.2 m long
LINE_SPACING = 30  # second doors: 3.0 m between door centres on one wall line
SEGMENT_SPACING = 32  # second doors: 2.0 m of wall to another opening on the same segment
LANDMARK_DEPTH = 1.5  # 0.15 m, in units
LANDMARK_WIDTHS = (0.3, 0.6, 1.0)
CLEARANCE_GAP_M = 0.5  # pilasters keep 0.5 m from doors, junctions and face ends
C = -1  # the corridor network's node in the door graph
# setting j: floor W × H (m), rooms R, doors per room d
SETTINGS = {
    1: (30, 20, 8, 1),
    2: (30, 20, 8, 2),
    3: (30, 20, 20, 1),
    4: (30, 20, 20, 2),
    5: (50, 30, 8, 1),
    6: (50, 30, 8, 2),
    7: (50, 30, 20, 1),
    8: (50, 30, 20, 2),
}
# planning ranges of plan §4.2 (central 95 % of 1,000 draws): partition length Λ_p (m), free area (m²)
RANGES = {
    1: ((35.2, 39.1), (574.0, 575.9)),
    2: ((35.2, 39.1), (575.9, 577.8)),
    3: ((107.3, 113.0), (562.2, 564.4)),
    4: ((107.3, 113.0), (567.0, 569.2)),
    5: ((27.6, 33.8), (1451.0, 1453.9)),
    6: ((27.6, 33.8), (1452.9, 1455.8)),
    7: ((144.8, 154.5), (1429.8, 1433.3)),
    8: ((144.8, 154.5), (1434.6, 1438.1)),
}


@dataclass(frozen=True, slots=True)
class Reviewed:
    """Values outside the planning ranges that the review found to be no generator error (plan §4.2); the
    waiver holds only while the measured values equal the reviewed ones."""

    partition_length_m: float
    free_area_m2: float
    note: str


RANGE_REVIEWED = {
    4: Reviewed(
        partition_length_m=108.8,
        free_area_m2=569.36,
        note="free area 569.36 m² is 0.16 m² above the range: only 3 pilasters (0.24 m²) fit, because 40 "
        "doors on the corridor walls drop the others; the free-area formula of §4.1 reproduces the built value "
        "(review, 18 October 2026)",
    ),
}
FAMILY_INFO = FamilyInfo(
    plan=PLAN,
    instances=tuple(f"{FAMILY}-{j:02d}" for j in SETTINGS),
    columns=4,
    seeds=seed_rules(SEED_BASE),
)

Rect = tuple[
    int, int, int, int
]  # clear extent x0, x1, y0, y1 in units from the floor's south-west corner


@dataclass(frozen=True, slots=True)
class Segment:
    """A stretch of wall between two spaces that may take a door: wall `axis` "x" runs along x at y = line."""

    a: int  # room index
    b: int  # room index or C
    axis: str
    line: int
    s0: int
    s1: int

    @property
    def length(self) -> int:
        return self.s1 - self.s0


@dataclass(frozen=True, slots=True)
class Opening:
    owner: int
    other: int
    segment: Segment
    centre: int


@dataclass(frozen=True, slots=True)
class Layout:
    width: int
    height: int
    spine: int  # spine centre y
    crosses: tuple[int, ...]  # cross corridor centres x
    blocks: tuple[Rect, ...]
    rooms: tuple[Rect, ...]
    room_block: tuple[int, ...]
    partitions: tuple[Rect, ...]  # partition wall rectangles
    segments: tuple[Segment, ...]
    openings: tuple[Opening, ...]
    passed_on: tuple[tuple[int, int], ...]


def _apportion(total: int, areas: list[int]) -> list[int]:
    """Largest-remainder apportionment of `total` by area with at least one per block.

    Blocks with a zero quota floor get one first; any shortfall or excess is then settled by the fractional
    remainders (largest first when adding, smallest first when removing; ties to the lower block index).
    """
    whole = sum(areas)
    quotas = [total * a / whole for a in areas]
    counts = [max(1, math.floor(q)) for q in quotas]
    remainders = [q - math.floor(q) for q in quotas]
    order = sorted(range(len(areas)), key=lambda i: (-remainders[i], i))
    i = 0
    while sum(counts) < total:
        counts[order[i % len(order)]] += 1
        i += 1
    order = sorted(range(len(areas)), key=lambda i: (remainders[i], i))
    i = 0
    while sum(counts) > total:
        b = order[i % len(order)]
        if counts[b] > 1:
            counts[b] -= 1
        i += 1
    return counts


def _split(region: Rect, axis: int, geom: np.random.Generator) -> tuple[Rect, Rect, Rect] | None:
    """Cut `region` across x (axis 0, a wall along y) or across y (axis 1); None when a child is too small."""
    x0, x1, y0, y1 = region
    start, end = (x0, x1) if axis == 0 else (y0, y1)
    t = float(geom.uniform(0.35, 0.65))
    centre = snap_units((start + t * (end - start)) * UNIT)
    lo, hi = centre - WALL // 2, centre + WALL // 2
    if lo - start < MIN_SIDE or end - hi < MIN_SIDE:
        return None
    if axis == 0:
        return (x0, lo, y0, y1), (hi, x1, y0, y1), (lo, hi, y0, y1)
    return (x0, x1, y0, lo), (x0, x1, hi, y1), (x0, x1, lo, hi)


def _subdivide(block: Rect, count: int, geom: np.random.Generator) -> tuple[list[Rect], list[Rect]]:
    regions = [block]
    walls: list[Rect] = []
    stuck: set[int] = set()
    while len(regions) < count:
        open_ = [i for i in range(len(regions)) if i not in stuck]
        if not open_:
            raise Infeasible("room_split_infeasible")
        idx = max(open_, key=lambda i: (_area(regions[i]), -i))
        x0, x1, y0, y1 = regions[idx]
        w, h = x1 - x0, y1 - y0
        if max(w, h) >= 1.25 * min(w, h):
            axis = 0 if w >= h else 1
        else:
            axis = int(geom.integers(2))
        result = _split(regions[idx], axis, geom)
        if result is None:
            result = _split(regions[idx], 1 - axis, geom)
        if result is None:
            stuck.add(idx)
            continue
        first, second, wall = result
        regions[idx] = first
        regions.append(second)
        walls.append(wall)
    return regions, walls


def _area(r: Rect) -> int:
    return (r[1] - r[0]) * (r[3] - r[2])


def _segments(rooms: list[Rect], spine: int, crosses: tuple[int, ...]) -> list[Segment]:
    """Room–room segments (overlap of at least 2.0 m across one wall) and room–corridor segments."""
    out: list[Segment] = []
    for i, a in enumerate(rooms):
        for j, b in enumerate(rooms):
            if j <= i:
                continue
            for p, q in ((a, b), (b, a)):
                lo, hi = max(p[2], q[2]), min(p[3], q[3])
                if q[0] - p[1] == WALL and hi - lo >= MIN_OVERLAP:
                    out.append(Segment(i, j, "y", p[1] + 1, lo, hi))
                lo, hi = max(p[0], q[0]), min(p[1], q[1])
                if q[2] - p[3] == WALL and hi - lo >= MIN_OVERLAP:
                    out.append(Segment(i, j, "x", p[3] + 1, lo, hi))
        x0, x1, y0, y1 = a
        if y1 == spine - CORRIDOR_HALF - WALL:
            out.append(Segment(i, C, "x", y1 + 1, x0, x1))
        if y0 == spine + CORRIDOR_HALF + WALL:
            out.append(Segment(i, C, "x", y0 - 1, x0, x1))
        for xc in crosses:
            if x1 == xc - CORRIDOR_HALF - WALL:
                out.append(Segment(i, C, "y", x1 + 1, y0, y1))
            if x0 == xc + CORRIDOR_HALF + WALL:
                out.append(Segment(i, C, "y", x0 - 1, y0, y1))
    return sorted(out, key=lambda s: (s.a, s.b, s.axis, s.line, s.s0))


def _other(seg: Segment, node: int) -> int:
    return seg.b if seg.a == node else seg.a


def _doors(
    n_rooms: int, segments: list[Segment], per_room: int, geom: np.random.Generator
) -> tuple[list[Opening], list[tuple[int, int]]]:
    """First doors (a tree) and, for d = 2, second doors; also the (skipped room, receiving room) pairs of
    second doors passed on (maintainer, 18 October 2026)."""
    usable = [s for s in segments if s.length >= MIN_SEGMENT]
    nbrs: dict[int, list[Segment]] = {n: [] for n in [C, *range(n_rooms)]}
    for s in usable:
        nbrs[s.a].append(s)
        nbrs[s.b].append(s)
    layer = {C: 0}
    todo = deque([C])
    while todo:
        node = todo.popleft()
        for s in nbrs[node]:
            other = _other(s, node)
            if other not in layer:
                layer[other] = layer[node] + 1
                todo.append(other)
    if len(layer) != n_rooms + 1:
        raise Infeasible("door_graph_disconnected")
    openings: list[Opening] = []
    for room in sorted(range(n_rooms), key=lambda r: (layer[r], r)):
        closer = sorted(
            {_other(s, room) for s in nbrs[room] if layer[_other(s, room)] == layer[room] - 1}
        )
        parent = closer[int(geom.choice(len(closer)))]
        segs = [s for s in nbrs[room] if _other(s, room) == parent]
        seg = segs[int(geom.choice(len(segs)))]
        centre = snap_units(
            float(geom.uniform((seg.s0 + DOOR_MARGIN) * UNIT, (seg.s1 - DOOR_MARGIN) * UNIT))
        )
        openings.append(Opening(room, parent, seg, centre))
    passed: list[tuple[int, int]] = []
    if per_room == 1:
        return openings, passed
    skipped = [room for room in range(n_rooms) if not _second_door(room, nbrs, openings, geom)]
    for room in skipped:
        for step in range(1, n_rooms):
            other = (room + step) % n_rooms
            if _second_door(other, nbrs, openings, geom):
                passed.append((room, other))
                break
        else:
            raise Infeasible("second_door_infeasible")
    return openings, passed


def _second_door(
    room: int, nbrs: dict[int, list[Segment]], openings: list[Opening], geom: np.random.Generator
) -> bool:
    """Add one opening owned by `room` (plan §3.4 step 9); False when no candidate class is non-empty."""
    allowed = {s: _allowed(s, openings) for s in nbrs[room]}
    allowed = {s: p for s, p in allowed.items() if p}
    linked = {_other(o.segment, room) for o in openings if room in (o.segment.a, o.segment.b)}
    nodes = sorted({_other(s, room) for s in allowed})
    classes = (
        [n for n in nodes if n != C and n not in linked],
        [n for n in nodes if n == C and C not in linked],
        nodes,
    )
    pool = next((c for c in classes if c), None)
    if pool is None:
        return False
    target = pool[int(geom.choice(len(pool)))]
    segs = [s for s in allowed if _other(s, room) == target]
    seg = segs[int(geom.choice(len(segs)))]
    positions = allowed[seg]
    openings.append(Opening(room, target, seg, positions[int(geom.integers(len(positions)))]))
    return True


def _allowed(seg: Segment, openings: list[Opening]) -> list[int]:
    """Door centres for a second door on `seg`: 1.1 m from its ends, 3.0 m from door centres on the same wall
    line and 3.2 m (2.0 m of wall) from openings on the same segment, on the 0.1 m grid."""
    taken = [
        (o.centre, SEGMENT_SPACING if o.segment == seg else LINE_SPACING)
        for o in openings
        if o.segment.axis == seg.axis and o.segment.line == seg.line
    ]
    return [
        p
        for p in range(seg.s0 + DOOR_MARGIN, seg.s1 - DOOR_MARGIN + 1)
        if all(abs(p - c) >= gap for c, gap in taken)
    ]


def _opening_rect(o: Opening) -> Rect:
    s = o.segment
    if s.axis == "x":
        return o.centre - DOOR // 2, o.centre + DOOR // 2, s.line - 1, s.line + 1
    return s.line - 1, s.line + 1, o.centre - DOOR // 2, o.centre + DOOR // 2


def _corridor_walls(width: int, height: int, spine: int, crosses: tuple[int, ...]) -> list[Rect]:
    """Corridor wall rectangles, interrupted where another corridor's clear width crosses them."""
    walls: list[Rect] = []
    ylo, yhi = spine - CORRIDOR_HALF - WALL, spine + CORRIDOR_HALF
    cuts = sorted((xc - CORRIDOR_HALF, xc + CORRIDOR_HALF) for xc in crosses)
    for y0 in (ylo, yhi):
        start = 0
        for a, b in cuts:
            walls.append((start, a, y0, y0 + WALL))
            start = b
        walls.append((start, width, y0, y0 + WALL))
    for xc in crosses:
        for x0 in (xc - CORRIDOR_HALF - WALL, xc + CORRIDOR_HALF):
            walls.append((x0, x0 + WALL, 0, spine - CORRIDOR_HALF))
            walls.append((x0, x0 + WALL, spine + CORRIDOR_HALF, height))
    return walls


def _layout(j: int, geom: np.random.Generator) -> Layout:
    w_m, h_m, n_rooms, per_room = SETTINGS[j]
    width, height = 10 * w_m, 10 * h_m
    offset = 1.5 if h_m == 20 else 2.0
    spine = height // 2 + snap_units(float(geom.uniform(-offset, offset)))
    if w_m == 30:
        crosses: tuple[int, ...] = (width // 2 + snap_units(float(geom.uniform(-5, 5))),)
    else:
        first = width // 2 + snap_units(-8.3 + float(geom.uniform(-3, 3)))
        crosses = (first, width // 2 + snap_units(8.3 + float(geom.uniform(-3, 3))))
    xs = [0]
    for xc in crosses:
        xs += [xc - CORRIDOR_HALF - WALL, xc + CORRIDOR_HALF + WALL]
    xs.append(width)
    ys = [(0, spine - CORRIDOR_HALF - WALL), (spine + CORRIDOR_HALF + WALL, height)]
    blocks = sorted(
        ((xs[2 * i], xs[2 * i + 1], y0, y1) for i in range(len(xs) // 2) for y0, y1 in ys),
        key=lambda r: (r[0], r[2]),
    )
    counts = _apportion(n_rooms, [_area(b) for b in blocks])
    rooms: list[Rect] = []
    room_block: list[int] = []
    partitions: list[Rect] = []
    for bi, (blk, count) in enumerate(zip(blocks, counts, strict=True)):
        regions, walls = _subdivide(blk, count, geom)
        rooms += regions
        room_block += [bi] * len(regions)
        partitions += walls
    segments = _segments(rooms, spine, crosses)
    openings, passed = _doors(n_rooms, segments, per_room, geom)
    return Layout(
        width,
        height,
        spine,
        crosses,
        tuple(blocks),
        tuple(rooms),
        tuple(room_block),
        tuple(partitions),
        tuple(segments),
        tuple(openings),
        tuple(passed),
    )


def _occupancy(lay: Layout) -> BoolArray:
    occ = np.zeros((lay.height, lay.width), dtype=bool)
    for x0, x1, y0, y1 in (
        *_corridor_walls(lay.width, lay.height, lay.spine, lay.crosses),
        *lay.partitions,
    ):
        occ[y0:y1, x0:x1] = True
    for o in lay.openings:
        x0, x1, y0, y1 = _opening_rect(o)
        occ[y0:y1, x0:x1] = False
    return occ


def _faces(lay: Layout) -> list[tuple[str, int, int, int, int, list[tuple[int, int]]]]:
    """Corridor wall faces in walk order: (axis, face line, inward sign, start, end, blocked intervals).

    Blocked intervals are the crossing corridors and the door openings on the face's wall line.
    """
    faces = []
    spine_lo, spine_hi = lay.spine - CORRIDOR_HALF, lay.spine + CORRIDOR_HALF
    cuts = [(xc - CORRIDOR_HALF, xc + CORRIDOR_HALF) for xc in lay.crosses]
    for line, wall_line, sign in ((spine_lo, spine_lo - 1, 1), (spine_hi, spine_hi + 1, -1)):
        doors = [
            (o.centre - DOOR // 2, o.centre + DOOR // 2)
            for o in lay.openings
            if o.segment.axis == "x" and o.segment.line == wall_line
        ]
        faces.append(("x", line, sign, 0, lay.width, cuts + doors))
    for xc in lay.crosses:
        for line, wall_line, sign in (
            (xc - CORRIDOR_HALF, xc - CORRIDOR_HALF - 1, 1),
            (xc + CORRIDOR_HALF, xc + CORRIDOR_HALF + 1, -1),
        ):
            doors = [
                (o.centre - DOOR // 2, o.centre + DOOR // 2)
                for o in lay.openings
                if o.segment.axis == "y" and o.segment.line == wall_line
            ]
            faces.append(("y", line, sign, 0, lay.height, [(spine_lo, spine_hi), *doors]))
    return faces


def _landmarks(lay: Layout, land: np.random.Generator) -> list[Primitive]:
    """Corridor pilasters (plan §3.4 step 11), in units converted to metres around the floor centre."""
    out: list[Primitive] = []
    gap = CLEARANCE_GAP_M / UNIT
    ox, oy = lay.width / 2, lay.height / 2
    for axis, line, sign, start, end, blocked in _faces(lay):
        pos = start + float(land.uniform(1.0, 4.0)) / UNIT
        while True:
            w = LANDMARK_WIDTHS[int(land.integers(3))] / UNIT
            centre = snap_units(pos * UNIT)
            a, b = centre - w / 2, centre + w / 2
            if b > end - gap:
                break
            clear = a >= start + gap and all(b <= lo - gap or a >= hi + gap for lo, hi in blocked)
            if clear:
                d0, d1 = sorted((line, line + sign * LANDMARK_DEPTH))
                name = f"landmark_{len(out):02d}"
                if axis == "x":
                    x0, x1, y0, y1 = a - ox, b - ox, d0 - oy, d1 - oy
                else:
                    x0, x1, y0, y1 = d0 - ox, d1 - ox, a - oy, b - oy
                out.append(
                    box(
                        name,
                        x0 * UNIT,
                        x1 * UNIT,
                        y0 * UNIT,
                        y1 * UNIT,
                        2.0,
                        Role.LANDMARK,
                        "landmark",
                    )
                )
            pos += float(land.uniform(4.0, 10.0)) / UNIT
    return out


@dataclass(frozen=True, slots=True)
class Candidate:
    world: World
    layout: Layout


def _candidate(j: int, seed: int, candidate: int, rejected: list[dict[str, object]]) -> Candidate:
    w_m, h_m, n_rooms, per_room = SETTINGS[j]
    geom, land, spawn = streams(seed)
    lay = _layout(j, geom)
    hx, hy = w_m / 2, h_m / 2
    outer, _ = walled_rectangle(-hx, hx, -hy, hy)
    walls = grid_walls(_occupancy(lay), UNIT, origin=(-hx, -hy))
    marks = _landmarks(lay, land)
    prims = (*outer, *walls, *marks)
    inclusion = ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    sx, sy, yaw = seeded_spawn(prims, (0.0, lay.spine * UNIT - hy), spawn)
    blocked = sum(p.size[0] * p.size[1] for p in (*walls, *marks))
    area = round(w_m * h_m - blocked, 4)
    partition_m = round(sum(max(x1 - x0, y1 - y0) for x0, x1, y0, y1 in lay.partitions) * UNIT, 2)
    doors: list[Door] = []
    for k, o in enumerate(lay.openings):
        x0, x1, y0, y1 = _opening_rect(o)
        cx, cy = (x0 + x1) / 2 * UNIT - hx, (y0 + y1) / 2 * UNIT - hy
        doors.append(door(f"D{k + 1:02d}", round(cx, 6), round(cy, 6), DOOR * UNIT, o.segment.axis))
    loops = (per_room - 1) * n_rooms
    world = World(
        id=f"{FAMILY}-{j:02d}",
        slug=f"office_f{w_m}x{h_m}_r{n_rooms:02d}_d{per_room}",
        title=f"Office {w_m}×{h_m} m, {n_rooms} rooms, {per_room} door{'s' if per_room > 1 else ''} per room",
        tier="F",
        role="evaluation",
        plan=PLAN,
        primitives=prims,
        spawns=(Spawn("S1", sx, sy, yaw, "seeded free cell, along the longest open axis ray"),),
        expected=Expected(
            free_area_m2=round(area, 2),
            narrowest_passage_m=1.2,
            loops=None,
            loops_unsized=loops,
            minimum_passage_m=1.2,
        ),
        family=FAMILY,
        parameters={
            "floor_m": [w_m, h_m],
            "rooms": n_rooms,
            "doors_per_room": per_room,
            "corridor_width_m": 2.0,
            "door_width_m": 1.2,
            "cross_corridors": len(lay.crosses),
        },
        seed=seed,
        doors=tuple(doors),
        dimensions=(
            Dimension((-hx, -hy - 0.9), (hx, -hy - 0.9), f"{w_m:.1f} m"),
            Dimension((-hx - 0.9, -hy), (-hx - 0.9, hy), f"{h_m:.1f} m"),
        ),
        inclusion=(inclusion,),
        sources=(
            {"what": "room segmentation in exploration", "cite": "holz2010evaluating"},
            {"what": "loops cause revisits", "cite": "xu2022explorebench"},
        ),
        extra_checks=(_checks(j, lay),),
        plan_data={
            "selection": {"candidate": candidate, "rejected": list(rejected)},
            "spine_y_m": round(lay.spine * UNIT - hy, 2),
            "cross_x_m": [round(xc * UNIT - hx, 2) for xc in lay.crosses],
            "rooms_per_block": [lay.room_block.count(b) for b in range(len(lay.blocks))],
            "partition_length_m": partition_m,
            "openings": len(lay.openings),
            "leaf_rooms": _leaf_rooms(lay),
            "second_doors_passed_on": [list(pair) for pair in lay.passed_on],
            "landmarks": len(marks),
            "construction_free_area_m2": area,
            "sheet_label": f"{n_rooms} rooms, {len(lay.openings)} doors, loops {loops}",
        },
    )
    return Candidate(world, lay)


def _leaf_rooms(lay: Layout) -> int:
    """Rooms with a single opening (owned or a neighbour's)."""
    count = [0] * len(lay.rooms)
    for o in lay.openings:
        for node in (o.segment.a, o.segment.b):
            if node != C:
                count[node] += 1
    return sum(1 for c in count if c == 1)


def _reject_reason(j: int, c: Candidate, a: Analysis) -> str | None:
    """Why candidate `c` of setting j is rejected during selection, or None to accept it.

    Selection runs the shared rules of `common.base_reject` (passages of at least 1.2 m, the unsized loop count
    (d − 1)·R) and the plan §7 family checks except the planning-range check, which only stops the build for
    review. The build runs everything else on the accepted candidate: `gz sdf -k`, the free-area tolerance,
    heights, determinism (T0 checks 1, 5, 6 and 8) and the planning-range check.
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


def _range_check(j: int, lay: Layout, a: Analysis) -> CheckResult:
    """Plan §4.2: partition length and free area inside the planning ranges, or equal to reviewed values."""
    (plo, phi), (alo, ahi) = RANGES[j]
    length = round(sum(max(x1 - x0, y1 - y0) for x0, x1, y0, y1 in lay.partitions) * UNIT, 2)
    area = round(a.free_area_m2, 2)
    inside = plo <= length <= phi and alo <= area <= ahi
    reviewed = RANGE_REVIEWED.get(j)
    waived = (
        not inside
        and reviewed is not None
        and reviewed.partition_length_m == length
        and reviewed.free_area_m2 == area
    )
    detail = f"ranges {RANGES[j]}"
    if waived and reviewed is not None:
        detail += f"; reviewed: {reviewed.note}"
    elif not inside and reviewed is not None:
        detail += (
            f"; differs from the reviewed values (partition {reviewed.partition_length_m} m, free area "
            f"{reviewed.free_area_m2} m²): {reviewed.note}"
        )
    return CheckResult(
        "F2 values inside the plan's planning ranges (outside: stop for review)",
        inside or waived,
        {"partition_length_m": length, "free_area_m2": area},
        detail,
    )


def _family_checks(
    j: int, lay: Layout, world: World, a: Analysis, *, ranges: bool
) -> list[CheckResult]:
    """The family checks of plan §7 on one candidate; `ranges` adds the planning-range check."""
    _, _, n_rooms, per_room = SETTINGS[j]
    hx, hy = lay.width / 2, lay.height / 2
    results: list[CheckResult] = []
    sides = [min(r[1] - r[0], r[3] - r[2]) * UNIT for r in lay.rooms]
    areas = [
        rect_area(a, *((v - o) * UNIT for v, o in zip(r, (hx, hx, hy, hy), strict=True)))
        for r in lay.rooms
    ]
    exact = [round(_area(r) * UNIT * UNIT, 4) for r in lay.rooms]
    results.append(
        CheckResult(
            "F2 rooms: R rooms, clear sides ≥ 2.5 m, interiors free",
            len(lay.rooms) == n_rooms
            and min(sides) >= 2.5 - 1e-9
            and all(abs(x - y) < 1e-6 for x, y in zip(areas, exact, strict=True)),
            {"rooms": len(lay.rooms), "min_side_m": round(min(sides), 2)},
        )
    )
    widths, _ = door_openings(a, world.doors)
    margins = [
        min(o.centre - DOOR // 2 - o.segment.s0, o.segment.s1 - o.centre - DOOR // 2) * UNIT
        for o in lay.openings
    ]
    same_line = [
        (p, q)
        for i, p in enumerate(lay.openings)
        for q in lay.openings[i + 1 :]
        if p.segment.axis == q.segment.axis and p.segment.line == q.segment.line
    ]
    wall_between = [
        ((abs(p.centre - q.centre) - DOOR) * UNIT, p.segment == q.segment) for p, q in same_line
    ]
    ok = (
        len(lay.openings) == per_room * n_rooms
        and all(abs(w - 1.2) <= a.grid.h + 1e-9 for w in widths)
        and min(margins) >= 0.5 - 1e-9
        and all(g >= (2.0 if same else 1.2) - 1e-9 for g, same in wall_between)
    )
    results.append(
        CheckResult(
            "F2 doors: d·R openings of 1.2 m, ≥ 0.5 m from junctions, spacing on wall lines",
            ok,
            {
                "openings": len(lay.openings),
                "widths_m": sorted(set(widths)),
                "min_margin_m": round(min(margins), 2),
                "min_wall_between_m": round(min((g for g, _ in wall_between), default=math.inf), 2),
            },
        )
    )
    parent = {n: n for n in [C, *range(n_rooms)]}

    def root(x: int) -> int:
        while parent[x] != x:
            x = parent[x]
        return x

    for o in lay.openings:
        parent[root(o.segment.a)] = root(o.segment.b)
    components = len({root(n) for n in parent})
    cycles = len(lay.openings) - (n_rooms + 1) + components
    results.append(
        CheckResult(
            "F2 door graph connected with (d − 1)·R cycles",
            components == 1 and cycles == (per_room - 1) * n_rooms,
            {"components": components, "cycles": cycles},
        )
    )
    corridor = []
    clear = []
    for xc in lay.crosses:
        corridor.append((xc - CORRIDOR_HALF, xc + CORRIDOR_HALF, 0, lay.height))
    corridor.append((0, lay.width, lay.spine - CORRIDOR_HALF, lay.spine + CORRIDOR_HALF))
    marks = [p for p in world.primitives if p.role is Role.LANDMARK]
    for x0, x1, y0, y1 in corridor:
        rect = ((x0 - hx) * UNIT, (x1 - hx) * UNIT, (y0 - hy) * UNIT, (y1 - hy) * UNIT)
        inside = sum(
            p.size[0] * p.size[1]
            for p in marks
            if rect[0] - 1e-6 <= p.x <= rect[1] + 1e-6 and rect[2] - 1e-6 <= p.y <= rect[3] + 1e-6
        )
        clear.append(
            abs(rect_area(a, *rect) - ((x1 - x0) * (y1 - y0) * UNIT * UNIT - inside)) < 1e-6
        )
    results.append(
        CheckResult(
            "F2 corridors 2.0 m clear wall to wall except pilasters",
            all(clear),
            clear,
        )
    )
    results.append(walls_check(FAMILY, world))
    if ranges:
        results.append(_range_check(j, lay, a))
    results.append(
        CheckResult(
            "F2 partitions = R − 2(k + 1)",
            len(lay.partitions) == n_rooms - 2 * (len(lay.crosses) + 1),
            len(lay.partitions),
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
    return {f"{FAMILY}-{j:02d}": functools.partial(build, j) for j in SETTINGS}
