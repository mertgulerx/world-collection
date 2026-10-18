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

"""World definition: primitives, spawns and annotations (T0 §4)."""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Final

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from .checks import Analysis
    from .moving import MovingDefinition

Point = tuple[float, float]

#: MOGI's circumscribed radius (m): the loop inflation of check 7 (T0 §7), the descriptors' robot and the E0
#: inflation of moving objects.
MOGI_RADIUS_M: Final = 0.29


class Role(StrEnum):
    BOUNDARY_WALL = "boundary wall"
    WALL = "wall"
    OBSTACLE = "obstacle"
    LANDMARK = "landmark"
    FURNITURE = "furniture"
    MOVING = "moving"
    VISUAL = "visual"


class Kind(StrEnum):
    BOX = "box"
    CYLINDER = "cylinder"
    POLYGON = "polygon"  # a vertical prism over a simple counter-clockwise polygon


#: Minimum height per role (T0 §7 check 5); None means exempt.
MIN_HEIGHT: dict[Role, float | None] = {
    Role.BOUNDARY_WALL: 2.0,
    Role.WALL: 2.0,
    Role.OBSTACLE: 1.0,
    Role.LANDMARK: 1.0,
    Role.FURNITURE: 1.0,
    Role.MOVING: None,
    Role.VISUAL: None,
}

#: Visual colours (RGB, 0-1). The lidar ignores colour.
MATERIALS: dict[str, tuple[float, float, float]] = {
    "floor": (0.86, 0.86, 0.84),
    "boundary": (0.62, 0.62, 0.62),
    "inner": (0.72, 0.66, 0.58),
    "core": (0.48, 0.48, 0.48),
    "obstacle": (0.40, 0.45, 0.52),
    "landmark": (0.20, 0.45, 0.80),
    "furniture": (0.55, 0.45, 0.35),
    "moving": (0.95, 0.55, 0.15),
}

NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")


@dataclass(frozen=True, slots=True)
class Primitive:
    """A static box, cylinder or polygon prism standing on the floor (z_min = 0 unless the role says otherwise).

    A polygon prism extrudes `vertices` (world frame, metres, counter-clockwise, simple, no holes) from z_min to
    z_min + height; (x, y) is its reference point (the SDF pose), its yaw is 0 and `size` is (height,).
    """

    kind: Kind
    name: str
    x: float
    y: float
    yaw: float
    size: tuple[float, ...]  # box: sx, sy, sz; cylinder: radius, height; polygon: height
    role: Role
    material: str
    z_min: float = 0.0
    vertices: tuple[Point, ...] = ()  # polygon prisms only

    @property
    def height(self) -> float:
        if self.kind is Kind.POLYGON:
            return self.size[0]
        return self.size[2] if self.kind is Kind.BOX else self.size[1]

    @property
    def z_max(self) -> float:
        return self.z_min + self.height

    def footprint_bounds(self) -> tuple[float, float, float, float]:
        """Axis-aligned bounds (xmin, xmax, ymin, ymax) of the footprint."""
        if self.kind is Kind.POLYGON:
            xs = [v[0] for v in self.vertices]
            ys = [v[1] for v in self.vertices]
            return min(xs), max(xs), min(ys), max(ys)
        if self.kind is Kind.CYLINDER:
            r = self.size[0]
            return self.x - r, self.x + r, self.y - r, self.y + r
        hx, hy = self.size[0] / 2, self.size[1] / 2
        c, s = abs(math.cos(self.yaw)), abs(math.sin(self.yaw))
        ex, ey = hx * c + hy * s, hx * s + hy * c
        return self.x - ex, self.x + ex, self.y - ey, self.y + ey


@dataclass(frozen=True, slots=True)
class Spawn:
    id: str
    x: float
    y: float
    yaw: float
    note: str = ""


@dataclass(frozen=True, slots=True)
class Notch:
    """A free-space landmark (recess or notch); drawn blue in the review images."""

    id: str
    polygon: tuple[Point, ...]


@dataclass(frozen=True, slots=True)
class Door:
    id: str
    x: float
    y: float
    width: float
    wall_axis: str  # "x" (wall runs along x) or "y"


@dataclass(frozen=True, slots=True)
class Dimension:
    """A dimension arrow for the plan image."""

    p0: Point
    p1: Point
    label: str


@dataclass(frozen=True, slots=True)
class Expected:
    free_area_m2: float
    narrowest_passage_m: float
    loops: int | None = None
    loops_unsized: int | None = None
    free_area_tolerance: float = 0.02
    minimum_passage_m: float = 1.0
    narrowest_tolerance_m: float | None = None  # None: one raster cell


@dataclass(frozen=True, slots=True)
class FamilyInfo:
    """A generated family (C1–C6): its instances in contact-sheet order and its seed rules."""

    plan: str
    instances: tuple[str, ...]
    columns: int
    seeds: dict[str, str]


@dataclass(frozen=True, slots=True)
class CheckResult:
    name: str
    passed: bool
    value: Any
    detail: str = ""


ExtraCheck = Callable[["World", "Analysis"], list[CheckResult]]


class PlanKey:
    """Keys of `World.plan_data` that the core modules read. Everything else in `plan_data` is a plan-specific
    record written to `world.yaml` as it is; these names are part of that layout, so they never change."""

    #: generated families: `{"candidate": i, "rejected": [...]}`; written as the top-level `selection` of
    #: `world.yaml` (not under `plan_data`) and summarised in `family.yaml`
    SELECTION: Final = "selection"
    #: `{route id: {PlanKey.POLYLINES_M: [[[x, y], ...], ...], ...}}`, drawn dashed on `plan.png` (R8, E worlds)
    PATROL_PATHS: Final = "patrol_paths"
    POLYLINES_M: Final = "polylines_m"
    #: the family contact sheet's label for an instance (default: loops and dead ends)
    SHEET_LABEL: Final = "sheet_label"
    #: the generator's own dead-end count, preferred on the contact sheet over T1's skeleton count (C1)
    DEAD_END_CELLS: Final = "dead_end_cells"
    #: the free area the generator built, which a world-specific check compares with the raster
    CONSTRUCTION_FREE_AREA_M2: Final = "construction_free_area_m2"


@dataclass(frozen=True, slots=True)
class World:
    id: str
    slug: str
    title: str
    tier: str
    role: str
    plan: str
    primitives: tuple[Primitive, ...]
    spawns: tuple[Spawn, ...]
    expected: Expected
    family: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    seed: int | None = None
    notches: tuple[Notch, ...] = ()
    doors: tuple[Door, ...] = ()
    dimensions: tuple[Dimension, ...] = ()
    inclusion: tuple[tuple[Point, ...], ...] | None = None  # None: derived from the raster (T0 §4)
    sources: tuple[dict[str, str], ...] = ()
    licence: str = (
        "Apache-2.0"  # maintainer, 18 October 2026: worlds carry the repository licence (report §9)
    )
    notice: str = ""  # NOTICE text written beside LICENSE (attribution of a layout source)
    extra_checks: tuple[ExtraCheck, ...] = ()
    plan_data: dict[str, Any] = field(default_factory=dict)
    structure_model: str | None = None  # variants keep their base world's name (T0 §6)
    plan_width_px: int = 1600  # review/plan.png width; 3,200 px for C5 and C6 (T0 §8)
    # side of review/plan_detail.png, a crop centred on S1 (C5, C6, R9)
    plan_detail_m: float | None = None
    # render_overview.png camera elevation; C6 uses 60° (its plan §3.5)
    overview_elevation_deg: float = 45.0
    # E worlds and PM1: the moving objects and the E0 system plugin (worldgen/moving.py); None for static worlds
    moving: MovingDefinition | None = None

    @property
    def structure_name(self) -> str:
        return self.structure_model or f"{self.id.replace('-', '_')}_structure"

    @property
    def folder(self) -> str:
        """`<ID>_<slug>`; a generated family instance (id `F1-01`) lives in `<family>_<slug>` (T0 §3)."""
        return f"{self.family}_{self.slug}" if self.family else f"{self.id}_{self.slug}"

    def validate(self) -> None:
        """Reject malformed definitions before anything is written."""
        names = [p.name for p in self.primitives]
        if len(names) != len(set(names)):
            dup = sorted({n for n in names if names.count(n) > 1})
            raise ValueError(f"{self.id}: duplicate primitive names {dup}")
        for p in self.primitives:
            if not NAME_PATTERN.match(p.name):
                raise ValueError(
                    f"{self.id}: primitive name {p.name!r} is not a lowercase identifier"
                )
            if p.material not in MATERIALS:
                raise ValueError(
                    f"{self.id}: primitive {p.name} has unknown material {p.material!r}"
                )
            values = (p.x, p.y, p.yaw, p.z_min, *p.size)
            if not all(math.isfinite(v) for v in values) or min(p.size) <= 0:
                raise ValueError(
                    f"{self.id}: primitive {p.name} has a non-finite or non-positive value"
                )
            expected_len = {Kind.BOX: 3, Kind.CYLINDER: 2, Kind.POLYGON: 1}[p.kind]
            if len(p.size) != expected_len:
                raise ValueError(f"{self.id}: primitive {p.name} needs {expected_len} size values")
            if p.kind is Kind.POLYGON:
                problem = polygon_problem(p.vertices)
                if problem or p.yaw != 0.0:
                    raise ValueError(
                        f"{self.id}: polygon {p.name} is invalid: {problem or 'yaw must be 0'}"
                    )
            elif p.vertices:
                raise ValueError(f"{self.id}: primitive {p.name} is not a polygon but has vertices")
            if p.role not in (Role.MOVING, Role.VISUAL) and p.z_min != 0.0:
                raise ValueError(f"{self.id}: primitive {p.name} must start at the floor")
        if not self.spawns:
            raise ValueError(f"{self.id}: at least one spawn is required")
        ids = [s.id for s in self.spawns]
        if len(ids) != len(set(ids)):
            raise ValueError(f"{self.id}: duplicate spawn ids")
        if self.moving is not None:
            self.moving.validate()

    def static_primitives(self) -> tuple[Primitive, ...]:
        return tuple(p for p in self.primitives if p.role not in (Role.MOVING, Role.VISUAL))

    def bounds(self) -> tuple[float, float, float, float]:
        boxes = [p.footprint_bounds() for p in self.primitives]
        return (
            min(b[0] for b in boxes),
            max(b[1] for b in boxes),
            min(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )


POLYGON_MIN_EDGE_M = 0.02


def polygon_area(vertices: Sequence[Point]) -> float:
    """Signed shoelace area (positive when counter-clockwise)."""
    n = len(vertices)
    return 0.5 * sum(
        vertices[i][0] * vertices[(i + 1) % n][1] - vertices[(i + 1) % n][0] * vertices[i][1]
        for i in range(n)
    )


def polygon_problem(vertices: Sequence[Point]) -> str:
    """Why the vertices are not a valid prism footprint, or "" when they are: at least three finite vertices
    rounded to 1 mm, no repeated closing vertex, every edge at least 2 cm long, counter-clockwise and simple
    (no two edges intersect except neighbours at their shared vertex)."""
    n = len(vertices)
    if n < 3:
        return "fewer than three vertices"
    pts = np.asarray(vertices, dtype=np.float64)
    if pts.shape != (n, 2) or not np.isfinite(pts).all():
        return "vertices are not finite (x, y) pairs"
    if not np.allclose(pts, np.round(pts, 3), rtol=0.0, atol=1e-9):
        return "vertices are not rounded to 1 mm"
    nxt = np.roll(pts, -1, axis=0)
    if float(np.hypot(*(nxt - pts).T).min()) < POLYGON_MIN_EDGE_M - 1e-9:
        return "an edge is shorter than 2 cm"
    if polygon_area(vertices) <= 0.0:
        return "not counter-clockwise"
    d = nxt - pts
    for i in range(n):
        a, b = pts[i], nxt[i]
        # orientation tests of edge i against every edge j, vectorised
        o1 = d[i, 0] * (pts[:, 1] - a[1]) - d[i, 1] * (pts[:, 0] - a[0])
        o2 = d[i, 0] * (nxt[:, 1] - a[1]) - d[i, 1] * (nxt[:, 0] - a[0])
        o3 = d[:, 0] * (a[1] - pts[:, 1]) - d[:, 1] * (a[0] - pts[:, 0])
        o4 = d[:, 0] * (b[1] - pts[:, 1]) - d[:, 1] * (b[0] - pts[:, 0])
        lo, hi = np.minimum(pts, nxt), np.maximum(pts, nxt)
        boxes = (
            (lo[:, 0] <= max(a[0], b[0]))
            & (hi[:, 0] >= min(a[0], b[0]))
            & (lo[:, 1] <= max(a[1], b[1]))
            & (hi[:, 1] >= min(a[1], b[1]))
        )
        hit = (o1 * o2 <= 0) & (o3 * o4 <= 0) & boxes
        hit[[i, (i - 1) % n, (i + 1) % n]] = False
        if hit.any():
            return f"not simple: edge {i} meets edge {int(np.argmax(hit))}"
    return ""


def sorted_primitives(primitives: tuple[Primitive, ...]) -> tuple[Primitive, ...]:
    """Primitives sorted by role, then name (T0 §4), so the same definition writes the same bytes."""
    order = {role: i for i, role in enumerate(Role)}
    return tuple(sorted(primitives, key=lambda p: (order[p.role], p.name)))


FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
