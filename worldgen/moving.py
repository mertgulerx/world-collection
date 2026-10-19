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

"""Moving-object path model, drawing engine, checks and writers.

The E0 gz-sim system (`world_collection/plugins/moving_objects/`) implements the same kinematics; this module is
the reference evaluator used by the checks and the review images. `python -m worldgen.agreement` compares a
recording of the system's pose topic with this evaluator (E0 §10, step 4).

Kinematics (E0 §3.4), with every rule the plan leaves open stated as the C++ system states it in
`plugins/moving_objects/src/Kinematics.hh`:

1. tau = 0 is at w_0 at the start of the move w_0 -> w_1.
2. Segments. loop: w_0->w_1, ..., w_{n-1}->w_0. patrol: the forward pass w_0->...->w_{n-1}, then the backward
   pass w_{n-1}->...->w_0.
3. Segment yaw. fixed mode: the object's `yaw`. heading mode: atan2(dy, dx) of the segment; a patrol backward
   segment w_k->w_{k-1} keeps the forward direction atan2(y_k - y_{k-1}, x_k - x_{k-1}) (reversal without turning).
4. Phases per segment, in this order: the move; on arrival at its end waypoint b, a turn (heading mode only)
   from this segment's yaw to the next segment's yaw (the next of the last segment is the first); then the
   dwell d_b. Turn delta = wrap(yaw_next - yaw_this), wrap(a) = math.remainder(a, 2 pi), plus 2 pi when the
   result is <= -pi, so delta lies in (-pi, pi]. A turn exists only when delta != 0. A loop therefore turns at
   w_0 at the end of the cycle (after w_{n-1}->w_0, before the d_0 dwell) and its yaw at tau = 0 is the heading
   of w_0->w_1. A patrol's endpoint deltas are exactly 0 by rule 3; its intermediate waypoints turn on both
   passes (the backward turn mirrors the forward one) and dwell on each arrival; its endpoints dwell once per
   cycle.
5. Durations are integer nanoseconds, each rounded once, half to even, on the exact double (Python's built-in
   round(), std::nearbyint in C++): move round(l / v * 1e9) with l = math.sqrt(dx * dx + dy * dy) (never
   hypot); turn round(abs(delta) / omega * 1e9); dwell round(d * 1e9). A phase that rounds to 0 ns is omitted.
   T is the sum; phase starts are the cumulative sums.
6. Pose at simulation time t: tau = ((t mod T) + phi) mod T in integers; the phase is the last whose start <=
   tau (binary search); f = float(tau - start) / float(duration). Move: x = x0 + f * (x1 - x0), y likewise,
   yaw = the segment yaw. Turn: the waypoint's position, yaw = yaw_this + f * delta (not re-wrapped). Dwell:
   the waypoint's position, yaw = yaw_next when delta != 0 at that waypoint, else yaw_this.
7. Path velocity: during a move v * (dx / l, dy / l); during a turn the yaw rate +-omega; zero in dwells. In
   `pose_and_velocity` mode the plugin commands p(t - dt) and the average velocity (p(t) - p(t - dt)) / dt, so the
   body ends each step at p(t); `velocity_at` and `expected_poses` report the path velocities above.

Written values are quantised before anything is evaluated (waypoints to 0.1 mm, dwells to 1 ms, speeds to
0.01 m/s, yaws to 1e-6 rad), so the values the plugin parses from the SDF are the doubles evaluated here.

Drawing engine (E0 §7), order of random draws from `rng.generator(seed, seed_key)`:

1. Candidates: the templates in declaration order; a convoy template with q slots gives q candidates.
2. One permutation of the candidates (`Generator.permutation`).
3. For each object index k: the next remaining candidate in permuted order (with `class_order`, the first
   remaining candidate of the first class in `class_order(k)` that has one). Each candidate is used once.
   - type: one uniform u, the first allowed type whose cumulative weight (renormalised over the template's
     allowed types, in their declared order) exceeds u;
   - instantiate the path for the type (the E module's callback); an empty path, or one shorter than the
     template's minimum, discards the draw and nothing more is drawn for it;
   - speed: one uniform u, v = floor((0.3 + 0.7 u) * 20 + 0.5) / 20 (U[0.3, 1.0] rounded to 0.05 m/s); a
     convoy slot after the convoy's first accepted member takes the convoy's speed and draws nothing;
   - phase: one uniform u, phi_ns = floor(floor(u * T_ns) / 1e8) * 1e8 (U[0, T) rounded down to 0.1 s); a
     later convoy slot j takes phi_j = (phi_first + floor((j - j_first) * T_ns / q)) mod T_ns and draws nothing.
     A later convoy slot whose own cycle differs from the convoy's (another yaw mode turns at the corners)
     cannot keep the spacing and is discarded;
   - accept if the checks M1-M6 pass for objects 0..k, else discard.
   Running out of candidates raises `DrawError`.
"""

from __future__ import annotations

import bisect
import math
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from .checks import Analysis
from .raster import EIGHT, Grid
from .rng import generator
from .sdf import m, rad
from .world import (
    MATERIALS,
    MOGI_RADIUS_M,
    BoolArray,
    CheckResult,
    Door,
    ExtraCheck,
    FloatArray,
    Kind,
    Primitive,
    Role,
    World,
)

IntArray = npt.NDArray[np.int64]

SYSTEM_NAME = "benchmark::MovingObjects"
PLUGIN_FILE = "benchmark-moving-objects-system"
FORMAT_VERSION = 1
TOPIC = "/benchmark/moving_objects/pose"
PUBLISH_PERIOD_MS = 50
TURN_RATE_RAD_S = 1.0
FLOOR_GAP_M = 0.01  # collision starts 0.01 m above the floor (E0 §5)
SPEED_RANGE = (0.3, 1.0)
MIN_SEGMENT_M = 0.05
MAX_DWELL_S = 30.0
PHASE_STEP_NS = 100_000_000  # phases are drawn rounded down to 0.1 s
NS = 1_000_000_000
DENSITY_AREA_M2 = {"low": 250.0, "high": 60.0}
OBJECT_ID = re.compile(r"^mo_\d{2}$")
TWO_PI = 2.0 * math.pi


class Mode(StrEnum):
    LOOP = "loop"
    PATROL = "patrol"


class YawMode(StrEnum):
    FIXED = "fixed"
    HEADING = "heading"


class VelocityMode(StrEnum):
    POSE = "pose"
    POSE_AND_VELOCITY = "pose_and_velocity"


class PhaseKind(StrEnum):
    MOVE = "move"
    TURN = "turn"
    DWELL = "dwell"


# ---------------------------------------------------------------------------------------------------------
# Object types (E0 §5)


@dataclass(frozen=True, slots=True)
class ObjectType:
    """A moving-object type: a solid box or cylinder of the listed mass.

    `size` is (sx, sy, height) for a box, with sx along the body x axis (the direction of travel in heading
    mode), or (radius, height) for a cylinder.
    """

    name: str
    kind: Kind
    size: tuple[float, ...]
    mass_kg: float
    yaw_mode: YawMode

    @property
    def height(self) -> float:
        return self.size[-1]

    @property
    def radius(self) -> float:
        """Circumscribed radius r of the footprint (m)."""
        if self.kind is Kind.CYLINDER:
            return self.size[0]
        return math.sqrt(self.size[0] ** 2 + self.size[1] ** 2) / 2

    @property
    def half_extent(self) -> tuple[float, float]:
        """Footprint half sizes (along body x, along body y); a cylinder gives (r, r)."""
        if self.kind is Kind.CYLINDER:
            return self.size[0], self.size[0]
        return self.size[0] / 2, self.size[1] / 2

    @property
    def footprint_length(self) -> float:
        """The longest footprint dimension (m): the convoy spacing rule (E0 §7)."""
        hx, hy = self.half_extent
        return 2 * max(hx, hy)

    @property
    def z(self) -> float:
        """Height of the model origin, the geometric centre: 0.01 + height / 2."""
        return FLOOR_GAP_M + self.height / 2

    def inertia(self) -> tuple[float, float, float]:
        """(ixx, iyy, izz) of the solid primitive about its centre (kg m²)."""
        mass = self.mass_kg
        if self.kind is Kind.CYLINDER:
            r, h = self.size
            side = mass * (3 * r * r + h * h) / 12
            return side, side, mass * r * r / 2
        sx, sy, sz = self.size
        return (
            mass * (sy * sy + sz * sz) / 12,
            mass * (sx * sx + sz * sz) / 12,
            mass * (sx * sx + sy * sy) / 12,
        )


OBJECT_TYPES: dict[str, ObjectType] = {
    t.name: t
    for t in (
        ObjectType("cyl_s", Kind.CYLINDER, (0.25, 1.2), 40.0, YawMode.FIXED),
        ObjectType("cyl_l", Kind.CYLINDER, (0.40, 1.2), 80.0, YawMode.FIXED),
        ObjectType("box", Kind.BOX, (0.60, 0.60, 1.2), 60.0, YawMode.HEADING),
        ObjectType("cart", Kind.BOX, (1.20, 0.60, 1.0), 80.0, YawMode.HEADING),
        ObjectType("pallet_truck", Kind.BOX, (1.60, 0.80, 1.2), 200.0, YawMode.HEADING),
        ObjectType("vehicle", Kind.BOX, (4.20, 1.80, 1.5), 1200.0, YawMode.HEADING),
    )
}


# ---------------------------------------------------------------------------------------------------------
# Path definition and kinematics (E0 §3.4)


def _q(value: float, decimals: int) -> float:
    """The double the plugin parses from `value` written with `decimals` fixed decimals."""
    return float(f"{value:.{decimals}f}") + 0.0  # + 0.0 turns -0.0 into 0.0


@dataclass(frozen=True, slots=True)
class Waypoint:
    """A waypoint (m, world frame) with its dwell (s) on each arrival."""

    x: float
    y: float
    dwell_s: float = 0.0

    def quantised(self) -> Waypoint:
        return Waypoint(_q(self.x, 4), _q(self.y, 4), _q(self.dwell_s, 3))


@dataclass(frozen=True, slots=True)
class MovingObject:
    """One moving model `mo_NN` and its path; the values written to the SDF and to `world.yaml`."""

    id: str
    type: str
    mode: Mode
    yaw_mode: YawMode
    speed: float  # m/s, a multiple of 0.01
    phase_ns: int
    waypoints: tuple[Waypoint, ...]
    yaw: float = 0.0  # used in fixed yaw mode (rad)
    template: str = ""
    convoy_slot: int | None = None
    turn_rate: float = TURN_RATE_RAD_S

    @property
    def object_type(self) -> ObjectType:
        return OBJECT_TYPES[self.type]

    @property
    def radius(self) -> float:
        return self.object_type.radius

    @property
    def timeline(self) -> Timeline:
        return timeline(
            self.waypoints, self.mode, self.yaw_mode, self.speed, self.yaw, self.turn_rate
        )

    @property
    def cycle_ns(self) -> int:
        return self.timeline.cycle_ns

    def validate(self) -> None:
        """The plugin's Configure validation (E0 §4), plus the round trip of every written value."""
        where = f"moving object {self.id}"
        if not OBJECT_ID.match(self.id):
            raise ValueError(f"{where}: id must be mo_NN")
        if self.type not in OBJECT_TYPES:
            raise ValueError(f"{where}: unknown type {self.type!r}")
        values = (
            self.speed,
            self.yaw,
            self.turn_rate,
            *(v for w in self.waypoints for v in (w.x, w.y, w.dwell_s)),
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError(f"{where}: non-finite value")
        if not SPEED_RANGE[0] <= self.speed <= SPEED_RANGE[1] or self.speed != _q(self.speed, 2):
            raise ValueError(
                f"{where}: speed {self.speed} outside [0.3, 1.0] or not a multiple of 0.01"
            )
        if self.yaw != _q(self.yaw, 6) or not -math.pi < self.yaw <= math.pi:
            raise ValueError(f"{where}: yaw {self.yaw} not in (-pi, pi] with 6 decimals")
        if self.turn_rate <= 0 or self.turn_rate != _q(self.turn_rate, 6):
            raise ValueError(f"{where}: turn rate must be positive with 6 decimals")
        if len(self.waypoints) < 2:
            raise ValueError(f"{where}: needs at least two waypoints")
        for k, w in enumerate(self.waypoints):
            if w != w.quantised():
                raise ValueError(f"{where}: waypoint {k} is not quantised (0.1 mm, 1 ms)")
            if not 0.0 <= w.dwell_s <= MAX_DWELL_S:
                raise ValueError(f"{where}: waypoint {k} dwell outside [0, 30] s")
        for a, b in _segment_pairs(len(self.waypoints), self.mode):
            if not _length(self.waypoints[a], self.waypoints[b]) > MIN_SEGMENT_M:
                raise ValueError(f"{where}: segment w_{a} -> w_{b} is not longer than 0.05 m")
        if not 0 <= self.phase_ns < self.cycle_ns:
            raise ValueError(f"{where}: phase_ns {self.phase_ns} outside [0, {self.cycle_ns})")


@dataclass(frozen=True, slots=True)
class Phase:
    """One phase of a timeline: position (x0, y0) plus f·(dx, dy), yaw yaw0 plus f·dyaw, f ∈ [0, 1)."""

    kind: PhaseKind
    start_ns: int
    duration_ns: int
    x0: float
    y0: float
    dx: float
    dy: float
    yaw0: float
    dyaw: float
    vx: float
    vy: float
    yaw_rate: float


@dataclass(frozen=True, slots=True, eq=False)
class PhaseArrays:
    """The phase fields of a timeline as arrays, for the vectorised evaluator."""

    start: IntArray
    duration: IntArray
    x0: FloatArray
    y0: FloatArray
    dx: FloatArray
    dy: FloatArray
    yaw0: FloatArray
    dyaw: FloatArray


@dataclass(frozen=True, slots=True, eq=False)
class Timeline:
    """The phases of one cycle, their integer start times and the cycle time T (ns)."""

    phases: tuple[Phase, ...]
    starts_ns: tuple[int, ...]
    cycle_ns: int
    arrays: PhaseArrays = field(repr=False)


def wrap(angle: float) -> float:
    """Wrap to (-pi, pi]: IEEE remainder by 2 pi (exact, as std::remainder), plus 2 pi at <= -pi."""
    r = math.remainder(angle, TWO_PI)
    if r <= -math.pi:
        r += TWO_PI
    return r


def _length(a: Waypoint, b: Waypoint) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    return math.sqrt(dx * dx + dy * dy)  # never hypot (E0 §3.4)


def _segment_pairs(n: int, mode: Mode) -> list[tuple[int, int]]:
    """Waypoint index pairs of the segments in cycle order."""
    if mode is Mode.LOOP:
        return [(i, (i + 1) % n) for i in range(n)]
    return [(i, i + 1) for i in range(n - 1)] + [(i, i - 1) for i in range(n - 1, 0, -1)]


def pass_pairs(n: int, mode: Mode) -> list[tuple[int, int]]:
    """Waypoint index pairs of one pass: the closed loop, or the forward pass of a patrol."""
    return _segment_pairs(n, mode)[: n if mode is Mode.LOOP else n - 1]


_TimelineKey = tuple[tuple[Waypoint, ...], Mode, YawMode, float, float, float]
_TIMELINES: dict[_TimelineKey, Timeline] = {}


def timeline(
    waypoints: tuple[Waypoint, ...],
    mode: Mode,
    yaw_mode: YawMode,
    speed: float,
    yaw: float = 0.0,
    turn_rate: float = TURN_RATE_RAD_S,
) -> Timeline:
    """Build (and memoise) the timeline of a path, exactly as the plugin does at Configure."""
    key: _TimelineKey = (waypoints, mode, yaw_mode, speed, yaw, turn_rate)
    cached = _TIMELINES.get(key)
    if cached is not None:
        return cached
    if len(waypoints) < 2 or speed <= 0 or turn_rate <= 0:
        raise ValueError("a path needs two waypoints, a positive speed and a positive turn rate")
    heading = yaw_mode is YawMode.HEADING
    segments: list[tuple[int, int, float]] = []
    for a, b in _segment_pairs(len(waypoints), mode):
        if not heading:
            seg_yaw = yaw
        else:
            lo, hi = (
                (a, b) if b > a or mode is Mode.LOOP else (b, a)
            )  # patrol backward: forward direction
            seg_yaw = math.atan2(
                waypoints[hi].y - waypoints[lo].y, waypoints[hi].x - waypoints[lo].x
            )
        segments.append((a, b, seg_yaw))
    phases: list[Phase] = []
    t = 0

    def append(kind: PhaseKind, duration: int, *values: float) -> None:
        nonlocal t
        if duration <= 0:
            return
        phases.append(Phase(kind, t, duration, *values))
        t += duration

    for k, (a, b, seg_yaw) in enumerate(segments):
        next_yaw = segments[(k + 1) % len(segments)][2]
        wa, wb = waypoints[a], waypoints[b]
        dx, dy = wb.x - wa.x, wb.y - wa.y
        length = math.sqrt(dx * dx + dy * dy)
        if not length > MIN_SEGMENT_M:
            raise ValueError(f"segment w_{a} -> w_{b} is not longer than 0.05 m")
        append(
            PhaseKind.MOVE,
            round(length / speed * 1e9),
            wa.x,
            wa.y,
            dx,
            dy,
            seg_yaw,
            0.0,
            dx / length * speed,
            dy / length * speed,
            0.0,
        )
        yaw_after = seg_yaw
        if heading:
            delta = wrap(next_yaw - seg_yaw)
            if delta != 0.0:
                rate = turn_rate if delta > 0.0 else -turn_rate
                append(
                    PhaseKind.TURN,
                    round(abs(delta) / turn_rate * 1e9),
                    wb.x,
                    wb.y,
                    0.0,
                    0.0,
                    seg_yaw,
                    delta,
                    0.0,
                    0.0,
                    rate,
                )
                yaw_after = next_yaw
        append(
            PhaseKind.DWELL,
            round(wb.dwell_s * 1e9),
            wb.x,
            wb.y,
            0.0,
            0.0,
            yaw_after,
            0.0,
            0.0,
            0.0,
            0.0,
        )

    def floats(values: list[float]) -> FloatArray:
        return np.array(values, dtype=np.float64)

    arrays = PhaseArrays(
        np.array([p.start_ns for p in phases], dtype=np.int64),
        np.array([p.duration_ns for p in phases], dtype=np.int64),
        floats([p.x0 for p in phases]),
        floats([p.y0 for p in phases]),
        floats([p.dx for p in phases]),
        floats([p.dy for p in phases]),
        floats([p.yaw0 for p in phases]),
        floats([p.dyaw for p in phases]),
    )
    result = Timeline(tuple(phases), tuple(p.start_ns for p in phases), t, arrays)
    _TIMELINES[key] = result
    return result


@dataclass(frozen=True, slots=True)
class Pose:
    x: float
    y: float
    yaw: float


class TimedPath(Protocol):
    """A path with its timeline and phase: a `MovingObject`, or a `ParsedObject` read back from an SDF."""

    @property
    def timeline(self) -> Timeline: ...

    @property
    def phase_ns(self) -> int: ...


def _locate(obj: TimedPath, t_ns: int) -> tuple[Phase, float]:
    tl = obj.timeline
    tau = ((t_ns % tl.cycle_ns) + obj.phase_ns) % tl.cycle_ns
    ph = tl.phases[bisect.bisect_right(tl.starts_ns, tau) - 1]
    return ph, float(tau - ph.start_ns) / float(ph.duration_ns)


def pose_at(obj: TimedPath, t_ns: int) -> Pose:
    """Commanded pose of `obj` at simulation time `t_ns` (integer nanoseconds)."""
    ph, f = _locate(obj, t_ns)
    return Pose(ph.x0 + f * ph.dx, ph.y0 + f * ph.dy, ph.yaw0 + f * ph.dyaw)


def velocity_at(obj: TimedPath, t_ns: int) -> tuple[float, float, float]:
    """Path velocity (vx, vy in the world frame, m/s; yaw rate, rad/s) at `t_ns`: the velocity of the path model
    (rule 7), not the average velocity the plugin commands over a step in `pose_and_velocity` mode."""
    ph, _ = _locate(obj, t_ns)
    return ph.vx, ph.vy, ph.yaw_rate


def pose_series(obj: TimedPath, times_ns: IntArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Vectorised `pose_at` over an int64 array of times; the same IEEE operations as the scalar form."""
    tl = obj.timeline
    arr = tl.arrays
    tau = ((times_ns % tl.cycle_ns) + obj.phase_ns) % tl.cycle_ns
    idx = np.searchsorted(arr.start, tau, side="right") - 1
    f = (tau - arr.start[idx]).astype(np.float64) / arr.duration[idx].astype(np.float64)
    x: FloatArray = arr.x0[idx] + f * arr.dx[idx]
    y: FloatArray = arr.y0[idx] + f * arr.dy[idx]
    yaw: FloatArray = arr.yaw0[idx] + f * arr.dyaw[idx]
    return x, y, yaw


def path_length(waypoints: Sequence[Waypoint], mode: Mode) -> float:
    """Length of one pass: the closed loop, or the forward pass of a patrol (m)."""
    return sum(_length(waypoints[a], waypoints[b]) for a, b in pass_pairs(len(waypoints), mode))


# ---------------------------------------------------------------------------------------------------------
# Footprint geometry (E0 §8): exact point, disc and rectangle distances, vectorised over time


@dataclass(frozen=True, slots=True, eq=False)
class Track:
    """A footprint over the sampled horizon: a disc of radius hx (cylinders) or a rectangle of half sizes
    (hx, hy) rotated by yaw."""

    disc: bool
    hx: float
    hy: float
    radius: float
    x: FloatArray
    y: FloatArray
    c: FloatArray
    s: FloatArray

    def at(self, idx: IntArray) -> Track:
        return Track(
            self.disc,
            self.hx,
            self.hy,
            self.radius,
            self.x[idx],
            self.y[idx],
            self.c[idx],
            self.s[idx],
        )


def track(obj: MovingObject, times_ns: IntArray) -> Track:
    x, y, yaw = pose_series(obj, times_ns)
    t = obj.object_type
    hx, hy = t.half_extent
    return Track(t.kind is Kind.CYLINDER, hx, hy, t.radius, x, y, np.cos(yaw), np.sin(yaw))


def region_track(cx: float, cy: float, hx: float, hy: float, n: int) -> Track:
    """A fixed axis-aligned rectangle as a track of n samples (doorway regions)."""
    ones = np.ones(n)
    return Track(False, hx, hy, math.hypot(hx, hy), cx * ones, cy * ones, ones, 0.0 * ones)


def _point_rect(px: float | FloatArray, py: float | FloatArray, t: Track) -> FloatArray:
    dx, dy = px - t.x, py - t.y
    u = dx * t.c + dy * t.s
    v = -dx * t.s + dy * t.c
    ex = np.maximum(np.abs(u) - t.hx, 0.0)
    ey = np.maximum(np.abs(v) - t.hy, 0.0)
    result: FloatArray = np.sqrt(ex * ex + ey * ey)
    return result


def point_distance(px: float | FloatArray, py: float | FloatArray, t: Track) -> FloatArray:
    """Distance from points to the footprint (0 inside)."""
    if t.disc:
        dx, dy = px - t.x, py - t.y
        result: FloatArray = np.maximum(np.sqrt(dx * dx + dy * dy) - t.hx, 0.0)
        return result
    return _point_rect(px, py, t)


def _corners(t: Track) -> list[tuple[FloatArray, FloatArray]]:
    out = []
    for su, sv in ((1, 1), (-1, 1), (-1, -1), (1, -1)):
        u, v = su * t.hx, sv * t.hy
        out.append((t.x + u * t.c - v * t.s, t.y + u * t.s + v * t.c))
    return out


def _rects_overlap(a: Track, b: Track) -> BoolArray:
    """Separating-axis test of two rectangles (closed sets)."""
    dx, dy = b.x - a.x, b.y - a.y
    separated = np.zeros(np.shape(dx), dtype=bool)
    for nx, ny in ((a.c, a.s), (-a.s, a.c), (b.c, b.s), (-b.s, b.c)):
        ra = a.hx * np.abs(a.c * nx + a.s * ny) + a.hy * np.abs(-a.s * nx + a.c * ny)
        rb = b.hx * np.abs(b.c * nx + b.s * ny) + b.hy * np.abs(-b.s * nx + b.c * ny)
        separated |= np.abs(dx * nx + dy * ny) > ra + rb
    result: BoolArray = ~separated
    return result


def footprint_distance(a: Track, b: Track) -> FloatArray:
    """Exact distance between two footprints at each sample (0 when they touch or overlap)."""
    if a.disc and b.disc:
        dx, dy = b.x - a.x, b.y - a.y
        result: FloatArray = np.maximum(np.sqrt(dx * dx + dy * dy) - a.hx - b.hx, 0.0)
        return result
    if a.disc or b.disc:
        disc, rect = (a, b) if a.disc else (b, a)
        result = np.maximum(_point_rect(disc.x, disc.y, rect) - disc.hx, 0.0)
        return result
    best = np.full(np.shape(a.x), np.inf)
    for px, py in _corners(a):
        best = np.minimum(best, _point_rect(px, py, b))
    for px, py in _corners(b):
        best = np.minimum(best, _point_rect(px, py, a))
    result = np.where(_rects_overlap(a, b), 0.0, best)
    return result


def footprint_polygon(obj: MovingObject, pose: Pose) -> list[tuple[float, float]]:
    """Footprint outline for drawing (a cylinder as a 32-gon)."""
    t = obj.object_type
    if t.kind is Kind.CYLINDER:
        r = t.size[0]
        return [
            (pose.x + r * math.cos(a), pose.y + r * math.sin(a))
            for a in np.linspace(0, TWO_PI, 32, endpoint=False)
        ]
    hx, hy = t.half_extent
    c, s = math.cos(pose.yaw), math.sin(pose.yaw)
    return [
        (pose.x + u * c - v * s, pose.y + u * s + v * c)
        for u, v in ((hx, hy), (-hx, hy), (-hx, -hy), (hx, -hy))
    ]


# ---------------------------------------------------------------------------------------------------------
# Doorway regions, limits and the check context


@dataclass(frozen=True, slots=True)
class DoorwayRegion:
    """An axis-aligned doorway region (E0 §8): centre and half sizes (m)."""

    id: str
    cx: float
    cy: float
    hx: float
    hy: float

    @property
    def polygon(self) -> tuple[tuple[float, float], ...]:
        x0, x1, y0, y1 = self.cx - self.hx, self.cx + self.hx, self.cy - self.hy, self.cy + self.hy
        return ((x0, y0), (x1, y0), (x1, y1), (x0, y1))


def doorway_regions(
    doors: Sequence[Door], wall_thickness: float = 0.2, extension: float = 0.5
) -> tuple[DoorwayRegion, ...]:
    """The door gap rectangle (gap width × wall thickness) extended `extension` m on both sides along the wall
    normal, for each door of the base world (E0 §8). `Door` records no thickness: every worldgen wall builder
    uses 0.2 m, the default here."""
    regions = []
    for d in doors:
        along, across = d.width / 2, wall_thickness / 2 + extension
        hx, hy = (along, across) if d.wall_axis == "x" else (across, along)
        regions.append(DoorwayRegion(d.id, d.x, d.y, hx, hy))
    return tuple(regions)


@dataclass(frozen=True, slots=True)
class Limits:
    """Check horizon and pass limits (E0 §8, `world.yaml` keys in the comments)."""

    check_horizon_s: float = 7800.0  # check_horizon_s
    sample_dt_s: float = 0.1
    blockage_limit_s: float = 10.0  # blockage_limit_s
    doorway_dwell_limit_s: float = 12.0  # doorway_dwell_limit_s
    spawn_exclusion_t0_m: float = 2.0  # spawn_exclusion_t0_m
    spawn_clearance_always_m: float = 1.0  # spawn_clearance_always_m
    separation_m: float = 0.1
    static_margin_m: float = 0.1
    robot_radius_m: float = MOGI_RADIUS_M  # MOGI's circumscribed radius
    blocked_area_m2: float = 1.0
    path_sample_m: float = 0.05

    def times_ns(self) -> IntArray:
        step = round(self.sample_dt_s * 1e9)
        count = round(self.check_horizon_s / self.sample_dt_s) + 1
        result: IntArray = np.arange(count, dtype=np.int64) * step
        return result


@dataclass(frozen=True, slots=True, eq=False)
class CheckContext:
    """The static data the moving-object checks need: the base world's raster, clearance and spawns.

    `free_eroded` is F of M6: reachable cells whose distance-transform clearance exceeds the robot radius.
    """

    grid: Grid
    clearance: FloatArray
    free_eroded: BoolArray
    free_components: int
    spawns: tuple[tuple[str, float, float], ...]
    doorways: tuple[DoorwayRegion, ...]
    limits: Limits
    times_ns: IntArray


def check_context(
    world: World, analysis: Analysis, doorways: Sequence[DoorwayRegion], limits: Limits = Limits()
) -> CheckContext:
    """The check context of a (base) world from its T0 analysis. M6 runs on the analysis grid (0.05 m)."""
    free = analysis.reachable & (analysis.clearance > limits.robot_radius_m)
    _, n = ndimage.label(free, structure=EIGHT)
    return CheckContext(
        analysis.grid,
        analysis.clearance,
        free,
        int(n),
        tuple((s.id, s.x, s.y) for s in world.spawns),
        tuple(doorways),
        limits,
        limits.times_ns(),
    )


# ---------------------------------------------------------------------------------------------------------
# Checks M1-M6 (E0 §8), evaluated incrementally so the drawing engine can test objects 0..k cheaply


def longest_run(flags: BoolArray) -> int:
    """Length of the longest run of True samples."""
    if not flags.any():
        return 0
    padded = np.concatenate(([0], flags.astype(np.int8), [0]))
    edges = np.flatnonzero(np.diff(padded))
    return int((edges[1::2] - edges[::2]).max())


def footprint_cells(
    tr: Track, i: int, grid: Grid, robot_radius_m: float, pad: int = 0
) -> tuple[int, int, int, int, BoolArray]:
    """M6's B(t_i) cells of one footprint: cells whose centre lies within the robot radius of the footprint at
    sample i, as a mask over the window (r0, r1, c0, c1) clipped to the grid and padded by `pad` cells."""
    g, rr = grid, robot_radius_m
    x, y = float(tr.x[i]), float(tr.y[i])
    reach = tr.radius + rr + g.h
    c0 = max(0, math.floor((x - reach - g.x0) / g.h) - pad)
    c1 = min(g.nx, math.ceil((x + reach - g.x0) / g.h) + 1 + pad)
    r0 = max(0, math.floor((y - reach - g.y0) / g.h) - pad)
    r1 = min(g.ny, math.ceil((y + reach - g.y0) / g.h) + 1 + pad)
    xs = g.x0 + (np.arange(c0, c1) + 0.5) * g.h
    ys = g.y0 + (np.arange(r0, r1) + 0.5) * g.h
    dx = xs[None, :] - x
    dy = ys[:, None] - y
    mask: BoolArray
    if tr.disc:
        mask = dx * dx + dy * dy <= (tr.hx + rr) ** 2
    else:
        c, s = float(tr.c[i]), float(tr.s[i])
        ex = np.maximum(np.abs(dx * c + dy * s) - tr.hx, 0.0)
        ey = np.maximum(np.abs(-dx * s + dy * c) - tr.hy, 0.0)
        mask = ex * ex + ey * ey <= rr * rr
    return r0, r1, c0, c1, mask


def free_cells(tracks: Sequence[Track], i: int, ctx: CheckContext) -> BoolArray:
    """U(t_i) of M6: the eroded free cells F minus every footprint's cells at sample i."""
    u = ctx.free_eroded.copy()
    for tr in tracks:
        r0, r1, c0, c1, mask = footprint_cells(tr, i, ctx.grid, ctx.limits.robot_radius_m)
        u[r0:r1, c0:c1] &= ~mask
    return u


def _dilate8(mask: BoolArray) -> BoolArray:
    p = np.pad(mask, 1)
    out = np.zeros_like(mask)
    h, w = mask.shape
    for dr in range(3):
        for dc in range(3):
            out |= p[dr : dr + h, dc : dc + w]
    return out


@dataclass(frozen=True, slots=True)
class ObjectValues:
    """Single-object check values."""

    static_margin_m: float  # M1: min path-point clearance - r - 0.1
    spawn_t0_m: float | None  # M2
    spawn_any_m: float | None  # M3
    doorway_dwell_s: float | None  # M5, longest interval
    dwell_in_doorway: tuple[str, ...]  # M5, waypoints with dwell inside an inflated region


@dataclass(frozen=True, slots=True)
class Measured:
    """`moving_objects.measured` of `world.yaml` and the pass rules of M1-M6."""

    min_static_clearance_margin_m: float | None
    min_spawn_distance_t0_m: float | None
    min_spawn_distance_any_t_m: float | None
    min_object_separation_margin_m: float | None
    max_doorway_dwell_s: float | None
    max_blockage_s: float | None
    dwell_in_doorway: tuple[str, ...]
    free_components: int
    limits: Limits

    def as_yaml(self) -> dict[str, Any]:
        def r(v: float | None) -> float | None:
            return None if v is None else round(v, 4)

        return {
            "min_static_clearance_margin_m": r(self.min_static_clearance_margin_m),
            "min_spawn_distance_t0_m": r(self.min_spawn_distance_t0_m),
            "min_spawn_distance_any_t_m": r(self.min_spawn_distance_any_t_m),
            "min_object_separation_margin_m": r(self.min_object_separation_margin_m),
            "max_doorway_dwell_s": r(self.max_doorway_dwell_s),
            "max_blockage_s": r(self.max_blockage_s),
        }

    def results(self) -> list[CheckResult]:
        lim, eps = self.limits, 1e-9

        def ok_min(v: float | None, bound: float) -> bool:
            return v is None or v >= bound - eps

        out = [
            CheckResult(
                "M1 static clearance",
                ok_min(self.min_static_clearance_margin_m, 0.0),
                self.min_static_clearance_margin_m,
                "min over path points of clearance - r - 0.1 m",
            ),
            CheckResult(
                "M2 spawn exclusion at t = 0",
                ok_min(self.min_spawn_distance_t0_m, lim.spawn_exclusion_t0_m),
                self.min_spawn_distance_t0_m,
                f">= {lim.spawn_exclusion_t0_m} m",
            ),
            CheckResult(
                "M3 spawn clearance at all times",
                ok_min(self.min_spawn_distance_any_t_m, lim.spawn_clearance_always_m),
                self.min_spawn_distance_any_t_m,
                f">= {lim.spawn_clearance_always_m} m over {lim.check_horizon_s:g} s",
            ),
            CheckResult(
                "M4 object separation",
                ok_min(self.min_object_separation_margin_m, 0.0),
                self.min_object_separation_margin_m,
                f"min footprint distance - {lim.separation_m} m",
            ),
            CheckResult(
                "M5 doorway dwell",
                (
                    self.max_doorway_dwell_s is None
                    or self.max_doorway_dwell_s <= lim.doorway_dwell_limit_s + eps
                )
                and not self.dwell_in_doorway,
                self.max_doorway_dwell_s,
                f"<= {lim.doorway_dwell_limit_s} s; dwell waypoints in doorways: "
                f"{', '.join(self.dwell_in_doorway) or 'none'}",
            ),
            CheckResult(
                "M6 route blockage",
                self.max_blockage_s is not None
                and self.max_blockage_s <= lim.blockage_limit_s + eps,
                self.max_blockage_s,
                f"<= {lim.blockage_limit_s} s"
                if self.free_components == 1
                else f"F has {self.free_components} components (base world build error)",
            ),
        ]
        return out


class _Evaluator:
    """Accepted objects and the per-instant blockage state.

    M6 is exact and incremental. `single[i]` records that U(t_i) has at most one 8-connected component. When
    object k is added and the cells of U next to its dilated footprint (its 8-neighbour ring) form one
    8-connected set, every path through the footprint can go round it, so the component partition is kept
    (minus the footprint's cells): a single component stays single and the instant stays unblocked. Every
    other instant is labelled in full.
    """

    def __init__(self, ctx: CheckContext) -> None:
        self.ctx = ctx
        n = len(ctx.times_ns)
        self.objects: list[MovingObject] = []
        self.tracks: list[Track] = []
        self.single: BoolArray = np.full(n, ctx.free_components == 1, dtype=bool)
        self.blocked: BoolArray = np.zeros(n, dtype=bool)
        self.values: list[ObjectValues] = []
        self.min_separation: float | None = None
        g = ctx.grid
        self.cells_needed = math.ceil(ctx.limits.blocked_area_m2 / (g.h * g.h) - 1e-9)

    # single-object checks --------------------------------------------------------------------------------
    def object_values(self, obj: MovingObject, tr: Track) -> ObjectValues:
        ctx, lim = self.ctx, self.ctx.limits
        g = ctx.grid
        r = obj.radius
        worst = math.inf
        for a, b in pass_pairs(len(obj.waypoints), obj.mode):
            wa, wb = obj.waypoints[a], obj.waypoints[b]
            length = _length(wa, wb)
            steps = math.ceil(length / lim.path_sample_m - 1e-9)
            s = np.minimum(np.arange(steps + 1) * lim.path_sample_m, length) / length
            px, py = wa.x + s * (wb.x - wa.x), wa.y + s * (wb.y - wa.y)
            cols = np.floor((px - g.x0) / g.h).astype(np.int64)
            rows = np.floor((py - g.y0) / g.h).astype(np.int64)
            inside = (rows >= 0) & (rows < g.ny) & (cols >= 0) & (cols < g.nx)
            clear = np.zeros(len(px))
            clear[inside] = ctx.clearance[rows[inside], cols[inside]]
            worst = min(worst, float(clear.min()))
        static_margin = worst - r - lim.static_margin_m
        t0 = any_t = None
        for _, sx, sy in ctx.spawns:
            d = point_distance(sx, sy, tr)
            t0 = float(d[0]) if t0 is None else min(t0, float(d[0]))
            any_t = float(d.min()) if any_t is None else min(any_t, float(d.min()))
        dwell = None
        bad: list[str] = []
        n_t = len(ctx.times_ns)
        for region in ctx.doorways:
            reg = region_track(region.cx, region.cy, region.hx, region.hy, n_t)
            inside_region = footprint_distance(tr, reg) <= lim.robot_radius_m
            run = longest_run(inside_region) * lim.sample_dt_s
            dwell = run if dwell is None else max(dwell, run)
            for k, w in enumerate(obj.waypoints):
                if w.dwell_s > 0:
                    one = region_track(region.cx, region.cy, region.hx, region.hy, 1)
                    if float(point_distance(w.x, w.y, one)[0]) <= r + lim.robot_radius_m:
                        bad.append(f"{obj.id} w_{k} in {region.id}")
        return ObjectValues(static_margin, t0, any_t, dwell, tuple(bad))

    def pair_separation(self, tr: Track) -> float | None:
        """Minimum over accepted objects and the horizon of the footprint distance minus 0.1 m."""
        lim = self.ctx.limits
        best: float | None = None
        for other in self.tracks:
            dx, dy = tr.x - other.x, tr.y - other.y
            centre = np.sqrt(dx * dx + dy * dy) - tr.radius - other.radius
            near = np.flatnonzero(centre < lim.separation_m + 1e-6)
            value = float(
                centre.min()
            )  # circumscribed discs: a lower bound that is exact when apart
            if len(near):
                exact = footprint_distance(tr.at(near), other.at(near))
                value = max(value, float(exact.min()))
            margin = value - lim.separation_m
            best = margin if best is None else min(best, margin)
        return best

    # M6 --------------------------------------------------------------------------------------------------
    def _mask(self, tr: Track, i: int, pad: int = 0) -> tuple[int, int, int, int, BoolArray]:
        return footprint_cells(tr, i, self.ctx.grid, self.ctx.limits.robot_radius_m, pad)

    def _full(self, tracks: Sequence[Track], i: int) -> tuple[bool, bool]:
        """(single, blocked) at sample i by labelling U(t_i) in full."""
        labels, n = ndimage.label(free_cells(tracks, i, self.ctx), structure=EIGHT)
        if n <= 1:
            return True, False
        sizes = np.bincount(labels.ravel())[1:]
        cut_off = int(sizes.sum() - sizes.max())
        return False, cut_off >= self.cells_needed

    def _ring_ok(self, tr: Track, i: int, near: Sequence[Track]) -> bool:
        r0, r1, c0, c1, mask = self._mask(tr, i, pad=1)
        free = self.ctx.free_eroded[r0:r1, c0:c1].copy()
        for other in near:
            q0, q1, p0, p1, om = self._mask(other, i)
            a0, a1, b0, b1 = max(r0, q0), min(r1, q1), max(c0, p0), min(c1, p1)
            if a0 < a1 and b0 < b1:
                free[a0 - r0 : a1 - r0, b0 - c0 : b1 - c0] &= ~om[
                    a0 - q0 : a1 - q0, b0 - p0 : b1 - p0
                ]
        ring = _dilate8(mask) & ~mask & free
        if not ring.any():
            return True
        _, n = ndimage.label(ring, structure=EIGHT)
        return int(n) <= 1

    def blockage_with(self, tr: Track) -> tuple[BoolArray, BoolArray]:
        """(single, blocked) per sample once `tr` is added to the accepted tracks."""
        single = self.single.copy()
        blocked = self.blocked.copy()
        if self.ctx.free_components != 1:
            return single, blocked
        rr, h = self.ctx.limits.robot_radius_m, self.ctx.grid.h
        near_flags = []
        for other in self.tracks:
            dx, dy = tr.x - other.x, tr.y - other.y
            gap = tr.radius + other.radius + 2 * rr + 4 * h
            near_flags.append(dx * dx + dy * dy < gap * gap)
        everyone = [*self.tracks, tr]
        cache: dict[tuple[float, float, float, float], bool] = {}
        for i in range(len(single)):
            if single[i]:
                near = [self.tracks[j] for j, flags in enumerate(near_flags) if flags[i]]
                if near:
                    ok = self._ring_ok(tr, i, near)
                else:
                    key = (float(tr.x[i]), float(tr.y[i]), float(tr.c[i]), float(tr.s[i]))
                    hit = cache.get(key)
                    ok = self._ring_ok(tr, i, ()) if hit is None else hit
                    cache[key] = ok
                if ok:
                    continue
            single[i], blocked[i] = self._full(everyone, i)
        return single, blocked

    def commit(
        self,
        obj: MovingObject,
        tr: Track,
        values: ObjectValues,
        separation: float | None,
        state: tuple[BoolArray, BoolArray],
    ) -> None:
        self.objects.append(obj)
        self.tracks.append(tr)
        self.values.append(values)
        if separation is not None:
            self.min_separation = (
                separation if self.min_separation is None else min(self.min_separation, separation)
            )
        self.single, self.blocked = state

    def max_blockage_s(self, blocked: BoolArray | None = None) -> float | None:
        if self.ctx.free_components != 1:
            return None
        flags = self.blocked if blocked is None else blocked
        return longest_run(flags) * self.ctx.limits.sample_dt_s

    def measured(self) -> Measured:
        def low(vals: list[float | None]) -> float | None:
            present = [v for v in vals if v is not None]
            return min(present) if present else None

        dwells = [v.doorway_dwell_s for v in self.values if v.doorway_dwell_s is not None]
        return Measured(
            low([v.static_margin_m for v in self.values]),
            low([v.spawn_t0_m for v in self.values]),
            low([v.spawn_any_m for v in self.values]),
            self.min_separation,
            max(dwells) if dwells else None,
            self.max_blockage_s(),
            tuple(b for v in self.values for b in v.dwell_in_doorway),
            self.ctx.free_components,
            self.ctx.limits,
        )

    def try_add(self, obj: MovingObject) -> str | None:
        """Add `obj` if M1-M6 pass for the accepted objects plus `obj`; otherwise return the reason."""
        lim, eps = self.ctx.limits, 1e-9
        tr = track(obj, self.ctx.times_ns)
        v = self.object_values(obj, tr)
        if v.static_margin_m < -eps:
            return f"M1 static clearance margin {v.static_margin_m:.3f} m"
        if v.spawn_t0_m is not None and v.spawn_t0_m < lim.spawn_exclusion_t0_m - eps:
            return f"M2 spawn distance at t = 0 {v.spawn_t0_m:.3f} m"
        if v.spawn_any_m is not None and v.spawn_any_m < lim.spawn_clearance_always_m - eps:
            return f"M3 spawn distance {v.spawn_any_m:.3f} m"
        if v.doorway_dwell_s is not None and v.doorway_dwell_s > lim.doorway_dwell_limit_s + eps:
            return f"M5 doorway dwell {v.doorway_dwell_s:.1f} s"
        if v.dwell_in_doorway:
            return f"M5 {v.dwell_in_doorway[0]}"
        sep = self.pair_separation(tr)
        if sep is not None and sep < -eps:
            return f"M4 separation margin {sep:.3f} m"
        state = self.blockage_with(tr)
        block = self.max_blockage_s(state[1])
        if block is None:
            return f"M6 F has {self.ctx.free_components} components"
        if block > lim.blockage_limit_s + eps:
            return f"M6 blockage {block:.1f} s"
        self.commit(obj, tr, v, sep, state)
        return None

    def add_track(self, tr: Track) -> None:
        """Add a footprint to the M6 state only (no single-object or separation values)."""
        state = self.blockage_with(tr)
        self.tracks.append(tr)
        self.single, self.blocked = state

    def add_unconditionally(self, obj: MovingObject) -> None:
        tr = track(obj, self.ctx.times_ns)
        self.commit(
            obj, tr, self.object_values(obj, tr), self.pair_separation(tr), self.blockage_with(tr)
        )


def measure(objects: Sequence[MovingObject], ctx: CheckContext) -> Measured:
    """The M1-M6 values of a list of objects over the horizon (adds them one by one; exact)."""
    ev = _Evaluator(ctx)
    for obj in objects:
        ev.add_unconditionally(obj)
    return ev.measured()


@dataclass(frozen=True, slots=True, eq=False)
class BlockageState:
    """M6's per-sample state of a list of objects: their tracks, and whether U(t_i) is in one component
    (`single`) and whether at least 1.0 m² is cut off (`blocked`). Regional variants of M6 (E1-3, E3-2) need
    to label only the instants where `split` is set; every other instant has U(t_i) in one component."""

    tracks: tuple[Track, ...]
    single: BoolArray
    blocked: BoolArray

    @property
    def split(self) -> BoolArray:
        result: BoolArray = ~self.single
        return result


def blockage_state(objects: Sequence[MovingObject], ctx: CheckContext) -> BlockageState:
    """The M6 state of `objects` over the horizon, built incrementally exactly as M6 builds it."""
    ev = _Evaluator(ctx)
    for obj in objects:
        ev.add_track(track(obj, ctx.times_ns))
    return BlockageState(tuple(ev.tracks), ev.single, ev.blocked)


# ---------------------------------------------------------------------------------------------------------
# Drawing engine (E0 §7)

Instantiate = Callable[[ObjectType], Sequence[Waypoint] | None]


@dataclass(frozen=True, slots=True)
class Template:
    """A path template of an E plan.

    `instantiate` returns the waypoints for a type (lane offsets, lane ends and spawn splits depend on the type's
    radius), or None/empty to discard the draw. A loop template with `convoy_slots` q > 0 enters the candidate
    set as q candidates that share the drawn speed and keep T/q spacing. `draw_class` groups templates for an E
    plan's class order (E1 §3.2, E3 §3.2).
    """

    name: str
    mode: Mode
    allowed_types: tuple[str, ...]
    instantiate: Instantiate
    draw_class: str = "all"
    convoy_slots: int = 0
    min_length_m: float = 0.0
    fixed_yaw: float = 0.0


@dataclass(frozen=True, slots=True)
class DrawSpec:
    """Everything the drawing engine needs from an E plan."""

    seed: int
    templates: tuple[Template, ...]
    type_weights: tuple[tuple[str, float], ...]
    seed_key: tuple[int, ...] = ()
    # object index k -> classes in order of preference; None: one class, the permuted order
    class_order: Callable[[int], Sequence[str]] | None = None


@dataclass(frozen=True, slots=True)
class Rejection:
    candidate: str
    type: str
    reason: str


@dataclass(frozen=True, slots=True)
class DrawResult:
    objects: tuple[MovingObject, ...]
    rejections: tuple[Rejection, ...]
    unused_candidates: int
    measured: Measured  # of all drawn objects (the high density)


class DrawError(RuntimeError):
    """Candidates ran out before the target count was reached (E0 §7): revise the E plan's templates."""

    def __init__(self, message: str, partial: DrawResult) -> None:
        super().__init__(message)
        self.partial = partial


def object_count(free_area_m2: float, area_per_object_m2: float) -> int:
    """round_half_up(A / a), at least 1 (E0 §6)."""
    return max(1, math.floor(free_area_m2 / area_per_object_m2 + 0.5))


def draw_speed(u: float) -> float:
    """U[0.3, 1.0] rounded to 0.05 m/s; k / 20 is the double nearest the written two-decimal value."""
    return math.floor((SPEED_RANGE[0] + (SPEED_RANGE[1] - SPEED_RANGE[0]) * u) * 20 + 0.5) / 20


def draw_phase_ns(u: float, cycle_ns: int) -> int:
    """U[0, T) rounded down to 0.1 s, in integer nanoseconds."""
    return (math.floor(u * cycle_ns) // PHASE_STEP_NS) * PHASE_STEP_NS


def _pick_type(template: Template, weights: Mapping[str, float], u: float) -> str:
    missing = [t for t in template.allowed_types if t not in weights]
    if missing:
        raise ValueError(f"template {template.name}: no type weight for {missing}")
    w = [weights[t] for t in template.allowed_types]
    total = sum(w)
    acc = 0.0
    for name, wi in zip(template.allowed_types, w, strict=True):
        acc += wi / total
        if u < acc:
            return name
    return template.allowed_types[-1]


def _instantiate(template: Template, type_name: str) -> tuple[Waypoint, ...] | None:
    wps = template.instantiate(OBJECT_TYPES[type_name])
    if not wps or len(wps) < 2:
        return None
    quantised = tuple(w.quantised() for w in wps)
    if path_length(quantised, template.mode) < template.min_length_m:
        return None
    return quantised


def draw(spec: DrawSpec, ctx: CheckContext, n_high: int) -> DrawResult:
    """Draw n_high objects (the high density; the low density is its first n_low objects)."""
    weights = dict(spec.type_weights)
    candidates: list[tuple[Template, int | None]] = []
    for tpl in spec.templates:
        if tpl.convoy_slots and tpl.mode is not Mode.LOOP:
            raise ValueError(f"template {tpl.name}: only loops can carry convoys")
        if not set(tpl.allowed_types) <= set(OBJECT_TYPES):
            raise ValueError(f"template {tpl.name}: unknown types {tpl.allowed_types}")
        if tpl.convoy_slots:
            longest = max(OBJECT_TYPES[t].footprint_length for t in tpl.allowed_types)
            for t in tpl.allowed_types:
                wps = _instantiate(tpl, t)
                if wps and path_length(wps, Mode.LOOP) / tpl.convoy_slots <= longest + 2.0:
                    raise ValueError(
                        f"template {tpl.name}: convoy spacing {path_length(wps, Mode.LOOP) / tpl.convoy_slots:.2f} m "
                        f"does not exceed the longest footprint {longest} m + 2.0 m"
                    )
            candidates += [(tpl, j) for j in range(tpl.convoy_slots)]
        else:
            candidates.append((tpl, None))
    rng = generator(spec.seed, spec.seed_key)
    order = [candidates[int(i)] for i in rng.permutation(len(candidates))]
    ev = _Evaluator(ctx)
    rejections: list[Rejection] = []
    convoys: dict[str, tuple[float, int, int, int]] = {}  # name -> (speed, phase_ns, first slot, T)

    def result() -> DrawResult:
        return DrawResult(tuple(ev.objects), tuple(rejections), len(order), ev.measured())

    for k in range(n_high):
        while True:
            index = None
            classes = spec.class_order(k) if spec.class_order else None
            if classes is None:
                index = 0 if order else None
            else:
                for cls in classes:
                    index = next((i for i, (t, _) in enumerate(order) if t.draw_class == cls), None)
                    if index is not None:
                        break
            if index is None:
                raise DrawError(
                    f"candidates ran out after {k} of {n_high} objects ({len(rejections)} rejected draws)",
                    result(),
                )
            tpl, slot = order.pop(index)
            label = tpl.name if slot is None else f"{tpl.name}[{slot}]"
            type_name = _pick_type(tpl, weights, float(rng.random()))
            wps = _instantiate(tpl, type_name)
            if wps is None:
                rejections.append(
                    Rejection(label, type_name, "path empty or below the minimum length")
                )
                continue
            yaw_mode = OBJECT_TYPES[type_name].yaw_mode
            base = MovingObject(
                f"mo_{k:02d}",
                type_name,
                tpl.mode,
                yaw_mode,
                0.3,
                0,
                wps,
                tpl.fixed_yaw,
                tpl.name,
                slot,
            )
            convoy = convoys.get(tpl.name) if slot is not None else None
            if convoy is None:
                speed = draw_speed(float(rng.random()))
                cycle = replace(base, speed=speed).cycle_ns
                phase = draw_phase_ns(float(rng.random()), cycle)
            else:
                speed, phase_first, first, cycle = convoy
                if replace(base, speed=speed).cycle_ns != cycle:
                    rejections.append(
                        Rejection(label, type_name, "cycle differs from the convoy's")
                    )
                    continue
                assert slot is not None
                phase = (phase_first + ((slot - first) * cycle) // tpl.convoy_slots) % cycle
            obj = replace(base, speed=speed, phase_ns=phase)
            obj.validate()
            reason = ev.try_add(obj)
            if reason is not None:
                rejections.append(Rejection(label, type_name, reason))
                continue
            if slot is not None and convoy is None:
                convoys[tpl.name] = (speed, phase, slot, cycle)
            break
    return result()


# ---------------------------------------------------------------------------------------------------------
# The definition written to the SDF and world.yaml (E0 §4, §6)


@dataclass(frozen=True, slots=True)
class MovingDefinition:
    """The moving objects of one E world: the plugin's values and the `world.yaml` block (E0 §4, §6).

    `area_per_object_m2` and `free_area_m2` are None only for scratch worlds without a density (M7 is then not
    applicable). `base_world` names the catalogue world whose structure must stay byte-identical (M9).
    """

    density: str
    seed: int
    objects: tuple[MovingObject, ...]
    velocity_mode: VelocityMode
    area_per_object_m2: float | None = None
    free_area_m2: float | None = None
    seed_key: tuple[int, ...] = ()
    doorways: tuple[DoorwayRegion, ...] = ()
    base_world: str | None = None
    rejected_draws: int = 0
    turn_rate_rad_s: float = TURN_RATE_RAD_S
    topic: str = TOPIC
    publish_period_ms: int = PUBLISH_PERIOD_MS
    limits: Limits = Limits()

    @property
    def count(self) -> int:
        return len(self.objects)

    def validate(self) -> None:
        if self.density not in ("none", "low", "high"):
            raise ValueError(f"moving objects: unknown density {self.density!r}")
        if not self.objects:
            raise ValueError("moving objects: no objects")
        for k, obj in enumerate(self.objects):
            if obj.id != f"mo_{k:02d}":
                raise ValueError(f"moving objects: object {k} must be mo_{k:02d}")
            if obj.turn_rate != self.turn_rate_rad_s:
                raise ValueError(f"moving objects: {obj.id} turn rate differs from the system's")
            obj.validate()
        if self.publish_period_ms <= 0 or self.publish_period_ms % 10:
            raise ValueError("moving objects: publish period must be a positive multiple of 10 ms")


def low_density(high: MovingDefinition, free_area_m2: float) -> MovingDefinition:
    """The nested low density: the first n_low objects of the high density, unchanged (E0 §6)."""
    n_low = object_count(free_area_m2, DENSITY_AREA_M2["low"])
    return replace(
        high,
        density="low",
        area_per_object_m2=DENSITY_AREA_M2["low"],
        objects=high.objects[:n_low],
    )


def variant(
    base: World,
    *,
    world_id: str,
    slug: str,
    title: str,
    moving: MovingDefinition,
    family: str,
    extra_checks: tuple[ExtraCheck, ...],
) -> World:
    """An E world: the base world's static structure, spawns and doors with moving objects added. The structure
    model keeps the base world's name (M9); `family` names the plan's folder `<family>_<slug>` (T0 §3)."""
    return replace(
        base,
        id=world_id,
        slug=slug,
        title=title,
        tier="MO",
        family=family,
        seed=moving.seed,
        structure_model=base.structure_name,
        moving=moving,
        extra_checks=extra_checks,
    )


def _object_type_check(obj: MovingObject) -> None:
    if obj.yaw_mode is not obj.object_type.yaw_mode:
        raise ValueError(f"{obj.id}: yaw mode differs from its type's (E0 §5)")


def sdf_plugin(defn: MovingDefinition, indent: str = "    ") -> str:
    """The `<plugin>` element of the system (E0 §4), indented for the `<world>` level."""
    i1, i2, i3 = indent, indent + "  ", indent + "    "
    lines = [
        f"{i1}<!-- E0 moving-object system; values written by worldgen -->",
        f'{i1}<plugin filename="{PLUGIN_FILE}" name="{SYSTEM_NAME}">',
        f"{i2}<format_version>{FORMAT_VERSION}</format_version>",
        f"{i2}<topic>{defn.topic}</topic>",
        f"{i2}<publish_period_ms>{defn.publish_period_ms}</publish_period_ms>",
        f"{i2}<velocity_mode>{defn.velocity_mode.value}</velocity_mode>",
        f"{i2}<turn_rate>{rad(defn.turn_rate_rad_s)}</turn_rate>",
        f"{i2}<seed>{defn.seed}</seed>",
    ]
    for obj in defn.objects:
        lines += [
            f"{i2}<object>",
            f"{i3}<model>{obj.id}</model>",
            f"{i3}<mode>{obj.mode.value}</mode>",
            f"{i3}<speed>{obj.speed:.2f}</speed>",
            f"{i3}<phase_ns>{obj.phase_ns}</phase_ns>",
            f"{i3}<cycle_ns>{obj.cycle_ns}</cycle_ns>",
            f"{i3}<yaw_mode>{obj.yaw_mode.value}</yaw_mode>",
            f"{i3}<yaw>{rad(obj.yaw)}</yaw>",
            f"{i3}<z>{m(obj.object_type.z)}</z>",
        ]
        lines += [
            f'{i3}<waypoint x="{m(w.x)}" y="{m(w.y)}" dwell="{w.dwell_s:.3f}"/>'
            for w in obj.waypoints
        ]
        lines.append(f"{i2}</object>")
    lines.append(f"{i1}</plugin>")
    return "\n".join(lines) + "\n"


def _geometry(t: ObjectType) -> str:
    if t.kind is Kind.BOX:
        sx, sy, sz = t.size
        return f"<box><size>{m(sx)} {m(sy)} {m(sz)}</size></box>"
    r, h = t.size
    return f"<cylinder><radius>{m(r)}</radius><length>{m(h)}</length></cylinder>"


def sdf_models(defn: MovingDefinition) -> str:
    """One non-static model `mo_NN` per object (E0 §5), posed at p(0)."""
    r, g, b = MATERIALS["moving"]
    colour = f"{r:.2f} {g:.2f} {b:.2f} 1"
    out = []
    for obj in defn.objects:
        _object_type_check(obj)
        t = obj.object_type
        p = pose_at(obj, 0)
        ixx, iyy, izz = t.inertia()
        geometry = _geometry(t)
        out.append(
            f'    <model name="{obj.id}">\n'
            "      <static>false</static>\n"
            f"      <pose>{m(p.x)} {m(p.y)} {m(t.z)} 0 0 {rad(wrap(p.yaw))}</pose>\n"
            '      <link name="body">\n'
            "        <gravity>false</gravity>\n"
            "        <inertial>\n"
            f"          <mass>{t.mass_kg:.3f}</mass>\n"
            f"          <inertia><ixx>{ixx:.6f}</ixx><ixy>0</ixy><ixz>0</ixz><iyy>{iyy:.6f}</iyy>"
            f"<iyz>0</iyz><izz>{izz:.6f}</izz></inertia>\n"
            "        </inertial>\n"
            '        <collision name="body_col">\n'
            f"          <geometry>{geometry}</geometry>\n"
            "        </collision>\n"
            '        <visual name="body_vis">\n'
            f"          <geometry>{geometry}</geometry>\n"
            f"          <material><ambient>{colour}</ambient><diffuse>{colour}</diffuse>"
            "<specular>0.05 0.05 0.05 1</specular></material>\n"
            "        </visual>\n"
            "      </link>\n"
            "    </model>\n"
        )
    return "".join(out)


def ledger_lines(defn: MovingDefinition) -> str:
    return (
        f"  Moving objects (E0):\n"
        f"    {defn.count} non-static models mo_00..mo_{defn.count - 1:02d}; density {defn.density}, seed "
        f"{defn.seed}, velocity mode {defn.velocity_mode.value};\n"
        f"    moved by the {SYSTEM_NAME} system plugin ({PLUGIN_FILE}).\n"
    )


def primitives_at(defn: MovingDefinition, t_ns: int = 0) -> tuple[Primitive, ...]:
    """The objects as `moving` primitives at time t (for world.json and the renders)."""
    prims = []
    for obj in defn.objects:
        t = obj.object_type
        p = pose_at(obj, t_ns)
        prims.append(
            Primitive(
                t.kind, obj.id, p.x, p.y, wrap(p.yaw), t.size, Role.MOVING, "moving", FLOOR_GAP_M
            )
        )
    return tuple(prims)


def yaml_block(defn: MovingDefinition, measured: Measured | None) -> dict[str, Any]:
    """The `moving_objects` block of `world.yaml` (E0 §6)."""
    lim = defn.limits
    objects = []
    for obj in defn.objects:
        t = obj.object_type
        row: dict[str, Any] = {
            "id": obj.id,
            "type": obj.type,
            "size": [
                round(v, 4)
                for v in (
                    t.size if t.kind is Kind.BOX else (2 * t.size[0], 2 * t.size[0], t.size[1])
                )
            ],
            "radius_m": round(t.radius, 4),
            "mode": obj.mode.value,
            "yaw_mode": obj.yaw_mode.value,
            "yaw": round(obj.yaw, 6),
            "speed_m_s": round(obj.speed, 2),
            "phase_ns": obj.phase_ns,
            "cycle_ns": obj.cycle_ns,
            "waypoints": [
                [round(w.x, 4), round(w.y, 4), round(w.dwell_s, 3)] for w in obj.waypoints
            ],
            "template": obj.template,
        }
        if obj.convoy_slot is not None:
            row["convoy_slot"] = obj.convoy_slot
        objects.append(row)
    return {
        "system": SYSTEM_NAME,
        "format_version": FORMAT_VERSION,
        "density": defn.density,
        "area_per_object_m2": defn.area_per_object_m2,
        "free_area_m2": None if defn.free_area_m2 is None else round(defn.free_area_m2, 4),
        "count": defn.count,
        "rejected_draws": defn.rejected_draws,
        "seed": defn.seed,
        "seed_key": list(defn.seed_key),
        "velocity_mode": defn.velocity_mode.value,
        "turn_rate_rad_s": defn.turn_rate_rad_s,
        "topic": defn.topic,
        "publish_period_ms": defn.publish_period_ms,
        "check_horizon_s": lim.check_horizon_s,
        "blockage_limit_s": lim.blockage_limit_s,
        "doorway_dwell_limit_s": lim.doorway_dwell_limit_s,
        "spawn_exclusion_t0_m": lim.spawn_exclusion_t0_m,
        "spawn_clearance_always_m": lim.spawn_clearance_always_m,
        "doorways": [
            {"id": d.id, "polygon": [[round(x, 4), round(y, 4)] for x, y in d.polygon]}
            for d in defn.doorways
        ],
        "objects": objects,
        "measured": None if measured is None else measured.as_yaml(),
    }


# ---------------------------------------------------------------------------------------------------------
# Reading the SDF back, and the checks M7-M10


@dataclass(frozen=True, slots=True)
class ParsedObject:
    """One `<object>` of the plugin element as written (the SDF carries no object type)."""

    id: str
    mode: Mode
    yaw_mode: YawMode
    speed: float
    phase_ns: int
    cycle_ns: int  # the written cycle; `timeline.cycle_ns` is the recomputed one
    yaw: float
    z: float
    waypoints: tuple[Waypoint, ...]
    turn_rate: float

    @property
    def timeline(self) -> Timeline:
        """The timeline recomputed from the written values (raises ValueError for an invalid path)."""
        return timeline(
            self.waypoints, self.mode, self.yaw_mode, self.speed, self.yaw, self.turn_rate
        )


@dataclass(frozen=True, slots=True)
class ParsedPlugin:
    """The plugin element of an SDF as the system parses it."""

    format_version: int
    topic: str
    publish_period_ms: int
    velocity_mode: str
    turn_rate: float
    seed: int
    objects: tuple[ParsedObject, ...]


def objects_from_sdf(sdf_text: str) -> ParsedPlugin:
    """Parse the plugin element as the system does (ValueError when it is missing or incomplete)."""
    root = ET.fromstring(sdf_text)
    plugin = next((p for p in root.iter("plugin") if p.get("name") == SYSTEM_NAME), None)
    if plugin is None:
        raise ValueError(f"no {SYSTEM_NAME} plugin")

    def text(el: ET.Element, tag: str) -> str:
        child = el.find(tag)
        if child is None or child.text is None:
            raise ValueError(f"plugin: missing <{tag}>")
        return child.text.strip()

    turn_rate = float(text(plugin, "turn_rate"))
    objects = tuple(
        ParsedObject(
            text(el, "model"),
            Mode(text(el, "mode")),
            YawMode(text(el, "yaw_mode")),
            float(text(el, "speed")),
            int(text(el, "phase_ns")),
            int(text(el, "cycle_ns")),
            float(text(el, "yaw")),
            float(text(el, "z")),
            tuple(
                Waypoint(
                    float(w.get("x", "nan")),
                    float(w.get("y", "nan")),
                    float(w.get("dwell", "nan")),
                )
                for w in el.findall("waypoint")
            ),
            turn_rate,
        )
        for el in plugin.findall("object")
    )
    return ParsedPlugin(
        int(text(plugin, "format_version")),
        text(plugin, "topic"),
        int(text(plugin, "publish_period_ms")),
        text(plugin, "velocity_mode"),
        turn_rate,
        int(text(plugin, "seed")),
        objects,
    )


def _model_block(sdf_text: str, name: str) -> str | None:
    start = sdf_text.find(f'    <model name="{name}">\n')
    if start < 0:
        return None
    end = sdf_text.find("\n    </model>\n", start)
    return sdf_text[start : end + len("\n    </model>\n")] if end >= 0 else None


def run_checks(
    world: World, analysis: Analysis, sdf_text: str, block: Mapping[str, Any], measured: Measured
) -> list[CheckResult]:
    """M1-M10 for a world with moving objects; `block` is the written `moving_objects` block."""
    defn = world.moving
    assert defn is not None
    results = measured.results()
    # M7 count
    if defn.area_per_object_m2 is None:
        results.append(CheckResult("M7 count", True, defn.count, "no density: not applicable"))
    else:
        area = analysis.free_area_m2
        want = object_count(area, defn.area_per_object_m2)
        same_area = defn.free_area_m2 is not None and abs(defn.free_area_m2 - area) < 1e-6
        results.append(
            CheckResult(
                "M7 count",
                defn.count == want and same_area,
                defn.count,
                f"round_half_up({area:.2f} / {defn.area_per_object_m2:g}) = {want}; recorded A "
                f"{defn.free_area_m2}",
            )
        )
    # M8 consistency
    problems: list[str] = []
    plugin: ParsedPlugin | None = None
    try:
        plugin = objects_from_sdf(sdf_text)
    except (ValueError, ET.ParseError) as exc:
        results.append(CheckResult("M8 consistency", False, None, f"SDF plugin unreadable: {exc}"))
    if plugin is not None:
        expect = {
            "format_version": (plugin.format_version, block["format_version"]),
            "topic": (plugin.topic, block["topic"]),
            "publish_period_ms": (plugin.publish_period_ms, block["publish_period_ms"]),
            "velocity_mode": (plugin.velocity_mode, block["velocity_mode"]),
            "turn_rate": (plugin.turn_rate, block["turn_rate_rad_s"]),
            "seed": (plugin.seed, block["seed"]),
        }
        problems += [f"{k}: SDF {a!r} yaml {b!r}" for k, (a, b) in expect.items() if a != b]
        rows = {row["id"]: row for row in block["objects"]}
        if [o.id for o in plugin.objects] != list(rows):
            problems.append("object ids differ")
        for obj in plugin.objects:
            row = rows.get(obj.id)
            if row is None:
                continue
            wps = [[w.x, w.y, w.dwell_s] for w in obj.waypoints]
            pairs = {
                "mode": (obj.mode.value, row["mode"]),
                "yaw_mode": (obj.yaw_mode.value, row["yaw_mode"]),
                "yaw": (obj.yaw, row["yaw"]),
                "speed": (obj.speed, row["speed_m_s"]),
                "phase_ns": (obj.phase_ns, row["phase_ns"]),
                "cycle_ns": (obj.cycle_ns, row["cycle_ns"]),
                "waypoints": (wps, row["waypoints"]),
                "z": (obj.z, round(OBJECT_TYPES[row["type"]].z, 4)),
            }
            problems += [
                f"{obj.id} {k}: SDF {a!r} yaml {b!r}" for k, (a, b) in pairs.items() if a != b
            ]
            try:
                cycle = obj.timeline.cycle_ns
            except ValueError as exc:
                problems.append(f"{obj.id}: {exc}")
                continue
            if cycle != obj.cycle_ns:
                problems.append(f"{obj.id}: recomputed cycle {cycle} != written {obj.cycle_ns}")
            if not 0 <= obj.phase_ns < cycle:
                problems.append(f"{obj.id}: phase outside [0, T)")
        results.append(
            CheckResult("M8 consistency", not problems, len(problems), "; ".join(problems[:5]))
        )
    # M9 static structure unchanged
    if defn.base_world is None:
        results.append(
            CheckResult(
                "M9 static structure unchanged", True, None, "no base world: not applicable"
            )
        )
    else:
        from . import catalog
        from .sdf import write_sdf

        base = catalog.get(defn.base_world)
        base_sdf = write_sdf(base)
        diff = [
            name
            for name in ("ground_plane", base.structure_name)
            if _model_block(base_sdf, name) is None
            or _model_block(base_sdf, name) != _model_block(sdf_text, name)
        ]
        results.append(
            CheckResult(
                "M9 static structure unchanged",
                not diff,
                "identical" if not diff else diff,
                f"ground_plane and {base.structure_name} against {defn.base_world}",
            )
        )
    # M10 reference exclusion
    root = ET.fromstring(sdf_text)
    statics = {
        model.get("name", ""): (model.findtext("static") or "").strip()
        for model in root.iterfind("world/model")  # top-level models (the plugin also has <model>)
    }
    moving_bad = [o.id for o in defn.objects if statics.get(o.id) != "false"]
    structure_bad = [n for n, s in statics.items() if not OBJECT_ID.match(n) and s != "true"]
    results.append(
        CheckResult(
            "M10 reference exclusion",
            not moving_bad
            and not structure_bad
            and len([n for n in statics if OBJECT_ID.match(n)]) == defn.count,
            {
                "moving_non_static": defn.count - len(moving_bad),
                "static_models": len(statics) - defn.count,
            },
            "; ".join(
                [
                    *(f"{n} not non-static" for n in moving_bad),
                    *(f"{n} not static" for n in structure_bad),
                ]
            ),
        )
    )
    return results


# ---------------------------------------------------------------------------------------------------------
# Scratch world for the plugin's parse and agreement tests (E0 §10 steps 3-4)

TEST_WORLD_ID = "E0T"


def test_objects(turn_rate: float = TURN_RATE_RAD_S) -> tuple[MovingObject, ...]:
    """Three objects: a loop in heading mode with turns and dwells, a three-waypoint patrol in heading mode
    (turns at its middle waypoint on both passes), and a fixed-yaw cylinder on a loop."""
    loop = MovingObject(
        "mo_00",
        "box",
        Mode.LOOP,
        YawMode.HEADING,
        0.5,
        3_700_000_000,
        (
            Waypoint(-6.0, -6.0, 2.0),
            Waypoint(2.0, -6.0),
            Waypoint(2.0, 2.0, 1.5),
            Waypoint(-6.0, 2.0),
        ),
        turn_rate=turn_rate,
    )
    patrol = MovingObject(
        "mo_01",
        "cart",
        Mode.PATROL,
        YawMode.HEADING,
        0.65,
        12_300_000_000,
        (Waypoint(-7.0, 6.5, 1.0), Waypoint(1.5, 6.5, 2.5), Waypoint(6.5, 1.5, 3.0)),
        turn_rate=turn_rate,
    )
    cylinder = MovingObject(
        "mo_02",
        "cyl_l",
        Mode.LOOP,
        YawMode.FIXED,
        0.8,
        0,
        (Waypoint(5.0, -7.0, 1.0), Waypoint(7.0, -2.0), Waypoint(4.5, -3.5, 0.5)),
        yaw=0.5,
        turn_rate=turn_rate,
    )
    return loop, patrol, cylinder


def test_world(velocity_mode: VelocityMode = VelocityMode.POSE_AND_VELOCITY) -> World:
    """A 20 × 20 m walled floor with the three test objects (not in the catalogue; scratch use only)."""
    from .geometry import walled_rectangle
    from .world import Expected, Spawn

    walls, _ = walled_rectangle(-10.0, 10.0, -10.0, 10.0)
    defn = MovingDefinition("none", 0, test_objects(), velocity_mode)
    defn.validate()
    return World(
        id=TEST_WORLD_ID,
        slug="moving_test",
        title="Moving-object test floor",
        tier="P",
        role="pilot",
        primitives=tuple(walls),
        spawns=(Spawn("S1", -8.5, -8.5, 0.0, "corner, away from every path"),),
        expected=Expected(free_area_m2=400.0, narrowest_passage_m=20.0, loops=0, loops_unsized=0),
        moving=defn,
    )


def expected_poses(objects: Sequence[MovingObject], stamps_ns: Sequence[int]) -> dict[str, Any]:
    """Poses p(t) and path velocities (rule 7) of every object at each stamp, for the agreement test. The
    velocities are those of the path model, not the step-average velocities the plugin commands in
    `pose_and_velocity` mode."""
    out: dict[str, Any] = {"stamps_ns": [int(t) for t in stamps_ns], "objects": {}}
    for obj in objects:
        rows = []
        for t in stamps_ns:
            p = pose_at(obj, int(t))
            vx, vy, wz = velocity_at(obj, int(t))
            rows.append([p.x, p.y, obj.object_type.z, p.yaw, wrap(p.yaw), vx, vy, wz])
        out["objects"][obj.id] = {
            "columns": ["x", "y", "z", "yaw", "yaw_wrapped", "vx", "vy", "yaw_rate"],
            "cycle_ns": obj.cycle_ns,
            "phase_ns": obj.phase_ns,
            "rows": rows,
        }
    return out
