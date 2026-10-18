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

"""RP3 Santa Maria del Mar, El Born, Barcelona (plan docs/plans/worlds/R10-santa-maria-del-mar.md).

The medieval street grid around the basilica inside the border the maintainer drew on OpenStreetMap. Built only
from the stored files in `sources/RP3/` (no network access): the raw Overpass extract and the georeferenced
border, each checked against its SHA-256 before use. Building footprints are rasterised on a 0.1 m construction
grid, clipped to the border and cleaned up (plan §3); each solid region then becomes one extruded polygon whose
outline is the region's simplified contour (decision 41). Buildings are solid, 6.0 m high, the basilica 12.0 m
and in the landmark colour. The OSM passages through buildings are open 2.0 m wide (decision 42), everything
north-west of Carrer de la Princesa is outside the world (decision 43, recorded in border.json) and one narrow
dead-end stub at the east edge is filled (decision 44).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from matplotlib.path import Path as MplPath
from scipy import ndimage

from .. import raster
from ..checks import Analysis, no_symmetry
from ..families.common import AXIS_RAYS, longest_ray_heading
from ..geometry import prism
from ..world import (
    BoolArray,
    CheckResult,
    Expected,
    FloatArray,
    Kind,
    Point,
    Primitive,
    Role,
    Spawn,
    ExtraCheck,
    World,
    polygon_problem,
)

SOURCE_DIR = Path(__file__).resolve().parents[2] / "sources" / "RP3"
OSM_FILE = "overpass_r10.json"
OSM_SHA256 = "ba19374694a5673c2ed7f45197e53d8e81af8c586cbcae86b8ac994a65925e81"
BORDER_FILE = "border.json"
BORDER_SHA256 = "2c30a90c19612b609c6dbc28145048c31a8f9a3d3233ecbeef69e04c4dff39c9"

UNIT = 0.1  # construction grid (plan §3)
MARGIN_CELLS = 8  # cells around the border polygon on the construction grid
BUILDING_H = 6.0
BASILICA_H = 12.0
BASILICA_NAME = "Santa Maria del Mar"  # the building whose name ends with it (plan §3)
WALL_T = 0.2  # boundary wall: one box of this thickness per border edge, its inner face on the edge
WALL_H = 3.0
CLOSE_R_CELLS = 4  # step 1: closing of the occupancy with a 0.4 m radius disk
PASSAGE_CELLS = 10  # step 3: free space must hold a 1.0 m disk everywhere
ALLEY_MIN_M = 1.0  # a step-3 piece at least this long is recorded as a filled alley, shorter as a corner fillet
RESIDUAL_MAX_M = 3.0
ARCH_HALF_M = 1.0  # decision 42: OSM passages through buildings (arches) opened 2.0 m wide
# decision 44: the narrow dead-end stub at the east edge, filled at the maintainer's request (world metres)
STUB = ((126.9, 13.0), (129.3, 12.7), (134.5, 18.0), (132.5, 20.5), (130.5, 19.5))
STUB_REASON = "removed at the maintainer's request (decision 44)"
SLIVER_BAND_M = 0.35  # border slivers: solid cells this close that a 0.2 m disk opening removes
# construction cells (0.3 m) a solid region grows beyond the border, under the boundary wall, and into a
# touching region of the other kind (building or basilica), so simplification opens no sliver between them
EXT_CELLS = 3
DP_EPS_M = (
    0.12,
    0.08,
    0.05,
)  # façade simplification tolerances, tried in order until the polygon is valid
SPAWN_CLEAR_M = 0.5
SPAWN_FALLBACK_CLEAR_M = 0.7
STREET_BAND_M = 3.0  # "the same street": cells within this distance of the street's centreline
S2_INSET_M = 20.0
S3_RADIUS_M = 30.0


def _read(name: str, sha256: str) -> Any:
    """A stored source file, parsed with json only after its SHA-256 matches the recorded value."""
    data = (SOURCE_DIR / name).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise ValueError(f"RP3: {name} has SHA-256 {digest}, expected {sha256}")
    return json.loads(data)


@dataclass(frozen=True, slots=True)
class Projection:
    """Equirectangular at the origin latitude (x east, y north, metres), from border.json."""

    lat0: float
    lon0: float
    m_lat: float
    m_lon: float

    def xy(self, lat: float, lon: float) -> Point:
        return (lon - self.lon0) * self.m_lon, (lat - self.lat0) * self.m_lat


@dataclass(frozen=True, slots=True)
class Footprint:
    osm: str  # "way/123" or "relation/456"
    name: str
    outers: tuple[tuple[Point, ...], ...]
    inners: tuple[tuple[Point, ...], ...]


def _rings(members: Sequence[Sequence[tuple[float, float]]]) -> list[list[tuple[float, float]]]:
    """Closed rings joined from way pieces that share end coordinates (multipolygon assembly)."""
    pieces = [list(m) for m in members if len(m) >= 2]
    rings: list[list[tuple[float, float]]] = []
    while pieces:
        ring = pieces.pop(0)
        while ring[0] != ring[-1]:
            for i, p in enumerate(pieces):
                if p[0] == ring[-1]:
                    ring += p[1:]
                elif p[-1] == ring[-1]:
                    ring += p[-2::-1]
                else:
                    continue
                pieces.pop(i)
                break
            else:
                break  # an unclosed ring (outside the extract): dropped
        if ring[0] == ring[-1] and len(ring) >= 4:
            rings.append(ring)
    return rings


def _footprints(osm: dict[str, Any], proj: Projection) -> list[Footprint]:
    out: list[Footprint] = []
    for e in osm["elements"]:
        tags = e.get("tags", {})
        if "building" not in tags or e["type"] not in ("way", "relation"):
            continue
        if e["type"] == "way":
            geo = [(g["lat"], g["lon"]) for g in e["geometry"]]
            outer, inner = _rings([geo]), []
        else:
            parts: dict[str, list[list[tuple[float, float]]]] = {"outer": [], "inner": []}
            for m in e.get("members", []):
                if m.get("type") == "way" and "geometry" in m and m.get("role") in parts:
                    parts[m["role"]].append([(g["lat"], g["lon"]) for g in m["geometry"]])
            outer, inner = _rings(parts["outer"]), _rings(parts["inner"])
        out.append(
            Footprint(
                f"{e['type']}/{e['id']}",
                str(tags.get("name", "")),
                tuple(tuple(proj.xy(*p) for p in r) for r in outer),
                tuple(tuple(proj.xy(*p) for p in r) for r in inner),
            )
        )
    return out


@dataclass(frozen=True, slots=True)
class Construction:
    """The 0.1 m construction grid: cell (r, c) centre at ((c0 + c + 0.5)·UNIT, (r0 + r + 0.5)·UNIT)."""

    c0: int
    r0: int
    nx: int
    ny: int

    @property
    def origin(self) -> Point:
        return round(self.c0 * UNIT, 4), round(self.r0 * UNIT, 4)

    def centres(self) -> tuple[FloatArray, FloatArray]:
        xs: FloatArray = (self.c0 + np.arange(self.nx, dtype=np.float64) + 0.5) * UNIT
        ys: FloatArray = (self.r0 + np.arange(self.ny, dtype=np.float64) + 0.5) * UNIT
        return xs, ys

    def cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor(y / UNIT)) - self.r0, int(math.floor(x / UNIT)) - self.c0

    def centre(self, r: int, c: int) -> Point:
        return (self.c0 + c + 0.5) * UNIT, (self.r0 + r + 0.5) * UNIT

    def grid(self) -> raster.Grid:
        x0, y0 = self.origin
        return raster.Grid(x0, y0, self.nx, self.ny, UNIT)

    def polygon(self, ring: Sequence[Point]) -> BoolArray:
        """Cells whose centre lies inside the ring, tested only within its bounding box."""
        mask = np.zeros((self.ny, self.nx), dtype=bool)
        arr = np.asarray(ring, dtype=float)
        c_lo = max(0, int(math.floor(arr[:, 0].min() / UNIT)) - self.c0 - 1)
        c_hi = min(self.nx, int(math.ceil(arr[:, 0].max() / UNIT)) - self.c0 + 1)
        r_lo = max(0, int(math.floor(arr[:, 1].min() / UNIT)) - self.r0 - 1)
        r_hi = min(self.ny, int(math.ceil(arr[:, 1].max() / UNIT)) - self.r0 + 1)
        if c_lo >= c_hi or r_lo >= r_hi:
            return mask
        xs, ys = self.centres()
        gx, gy = np.meshgrid(xs[c_lo:c_hi], ys[r_lo:r_hi])
        inside = MplPath(arr).contains_points(np.column_stack([gx.ravel(), gy.ravel()]))
        mask[r_lo:r_hi, c_lo:c_hi] = inside.reshape(gy.shape)
        return mask


def _disk(radius_cells: float, even: bool = False) -> BoolArray:
    """A disk structuring element; `even` centres it between cells (extent 2·radius cells)."""
    if even:
        n = int(round(2 * radius_cells))
        k = np.arange(n) - (n - 1) / 2
    else:
        n = 2 * int(radius_cells) + 1
        k = np.arange(n) - (n - 1) / 2
    gx, gy = np.meshgrid(k, k)
    disk: BoolArray = gx * gx + gy * gy <= radius_cells * radius_cells + 1e-9
    return disk


def _components(mask: BoolArray) -> tuple[np.ndarray[Any, np.dtype[np.int32]], int]:
    labels, n = ndimage.label(mask)
    return labels, int(n)


def _largest_component(mask: BoolArray) -> tuple[BoolArray, int]:
    """The largest 4-connected component of the mask, and the number of components."""
    labels, n = _components(mask)
    sizes = np.bincount(labels.ravel())[1:]
    largest: BoolArray = labels == int(np.argmax(sizes)) + 1
    return largest, n


@dataclass(frozen=True, slots=True)
class Layout:
    con: Construction
    inside: BoolArray
    free: BoolArray
    building: BoolArray  # buildings and clean-up fill (role wall)
    basilica: BoolArray
    records: dict[str, Any]


def _cleanup(
    con: Construction, inside: BoolArray, solid: BoolArray
) -> tuple[BoolArray, dict[str, Any]]:
    """Plan §3 clean-up on the construction grid; returns the free cells and the recorded counts."""
    occupied = solid | ~inside
    # 1: closing with a 0.4 m radius disk (gaps narrower than about 0.8 m); outside the grid counts as occupied
    pad = CLOSE_R_CELLS + 1
    padded = np.pad(occupied, pad, constant_values=True)
    closed = ndimage.binary_closing(padded, structure=_disk(CLOSE_R_CELLS))[pad:-pad, pad:-pad]
    closed |= occupied
    step1 = closed & ~occupied
    labels1, n1 = _components(step1)
    # 2: free regions not connected to the main street network
    free = inside & ~closed
    main, n = _largest_component(free)
    step2 = free & ~main
    free = main
    # 3: passages narrower than 1.0 m: every free cell must lie in a 1.0 m disk inside the free space
    opened = ndimage.binary_opening(free, structure=_disk(PASSAGE_CELLS / 2, even=True))
    opened &= free
    narrow = free & ~opened
    labels3, n3 = _components(narrow)
    dist = ndimage.distance_transform_edt(free) * UNIT
    alleys: list[dict[str, Any]] = []
    fillets = 0
    for i, sl in enumerate(ndimage.find_objects(labels3), start=1):
        if sl is None:
            continue
        part = labels3[sl] == i
        side = max(sl[0].stop - sl[0].start, sl[1].stop - sl[1].start) * UNIT
        if side < ALLEY_MIN_M - 1e-9:
            fillets += 1
            continue
        rows, cols = np.nonzero(part)
        rr, cc = int(rows.mean()) + sl[0].start, int(cols.mean()) + sl[1].start
        x, y = con.centre(rr, cc)
        width = 2.0 * float(dist[sl][part].max()) - UNIT
        alleys.append(
            {
                "x": round(x, 1),
                "y": round(y, 1),
                "length_m": round(side, 1),
                "width_m": round(max(width, UNIT), 1),
                "area_m2": round(float(part.sum()) * UNIT * UNIT, 2),
            }
        )
    free = opened
    main4, n4 = _largest_component(free)
    step3_sealed = free & ~main4
    free = main4
    a = UNIT * UNIT
    records = {
        "step1_closing": {
            "radius_m": CLOSE_R_CELLS * UNIT,
            "pieces": n1,
            "filled_m2": round(float(step1.sum()) * a, 2),
        },
        "step2_unconnected": {
            "regions": n - 1,
            "filled_m2": round(float(step2.sum()) * a, 2),
        },
        "step3_narrow": {
            "minimum_m": PASSAGE_CELLS * UNIT,
            "alleys": len(alleys),
            "alley_locations": alleys,
            "corner_fillets": fillets,
            "pieces": n3,
            "filled_m2": round(float(narrow.sum()) * a, 2),
            "sealed_behind_regions": n4 - 1,
            "sealed_behind_m2": round(float(step3_sealed.sum()) * a, 2),
        },
    }
    return free, records


def _construction(border: dict[str, Any]) -> tuple[Construction, list[Point]]:
    """The construction grid around the border polygon (MARGIN_CELLS on every side) and the polygon."""
    poly = [(float(x), float(y)) for x, y in border["border_m"]]
    xs = [q[0] for q in poly]
    ys = [q[1] for q in poly]
    c0 = int(math.floor(min(xs) / UNIT)) - MARGIN_CELLS
    r0 = int(math.floor(min(ys) / UNIT)) - MARGIN_CELLS
    nx = int(math.ceil(max(xs) / UNIT)) + MARGIN_CELLS - c0
    ny = int(math.ceil(max(ys) / UNIT)) + MARGIN_CELLS - r0
    return Construction(c0, r0, nx, ny), poly


def _rasterise_footprints(
    feet: list[Footprint], con: Construction, inside: BoolArray
) -> tuple[BoolArray, BoolArray, list[str], str]:
    """Building and basilica cells inside the border (the basilica's cells only in its own mask), the OSM ids
    of the footprints that reach inside, and the basilica's OSM id."""
    building = np.zeros_like(inside)
    basilica = np.zeros_like(inside)
    used: list[str] = []
    basilica_ids: list[str] = []
    for f in feet:
        mask = np.zeros_like(inside)
        for ring in f.outers:
            mask |= con.polygon(ring)
        for ring in f.inners:
            mask &= ~con.polygon(ring)
        mask &= inside
        if not mask.any():
            continue
        used.append(f.osm)
        if f.name.endswith(BASILICA_NAME):
            basilica |= mask
            basilica_ids.append(f.osm)
        else:
            building |= mask
    if len(basilica_ids) != 1:
        raise ValueError(f"RP3: expected one basilica footprint, found {basilica_ids}")
    building &= ~basilica
    return building, basilica, used, basilica_ids[0]


def _post_cleanup(
    con: Construction, inside: BoolArray, free: BoolArray, basilica: BoolArray
) -> tuple[BoolArray, dict[str, Any]]:
    """After the plan §3 clean-up: border slivers opened, the stub filled (decision 44) with its narrow
    remnants, diagonal contacts filled, and the main component kept; returns the free cells and the records."""
    # solid slivers along the border thinner than about 0.4 m (raw cells of buildings across the Princesa
    # façade line) become street: they would only make teeth under the boundary wall
    solid = inside & ~free & ~basilica
    near = ndimage.distance_transform_edt(inside) * UNIT <= SLIVER_BAND_M + 1e-9
    sliver = solid & near & ~ndimage.binary_opening(solid, structure=_disk(2))
    free = free | sliver
    stub = free & con.polygon(STUB)
    free &= ~stub
    # the stub's remnants narrower than 1.0 m go too (the step-3 rule, applied again)
    reopened = ndimage.binary_opening(free, structure=_disk(PASSAGE_CELLS / 2, even=True)) & free
    stub |= free & ~reopened
    free = reopened
    free, pinches = _depinch(free, inside)
    free, _ = _largest_component(free)
    records = {
        "stub": {
            "polygon": [list(q) for q in STUB],
            "filled_m2": round(float(stub.sum()) * UNIT * UNIT, 2),
            "reason": STUB_REASON,
        },
        "diagonal_contacts_filled": pinches,
        "border_slivers_opened_m2": round(float(sliver.sum()) * UNIT * UNIT, 2),
    }
    return free, records


def _layout(
    osm: dict[str, Any], border: dict[str, Any]
) -> tuple[Layout, list[Footprint], Projection]:
    p = border["projection"]
    proj = Projection(p["origin_lat"], p["origin_lon"], p["m_per_deg_lat"], p["m_per_deg_lon"])
    con, poly = _construction(border)
    inside = con.polygon(poly)
    feet = _footprints(osm, proj)
    building, basilica, used, basilica_osm = _rasterise_footprints(feet, con, inside)
    raw_solid = building | basilica
    arches, arch_records = _arches(osm, proj, con, inside, raw_solid)
    building &= ~arches
    basilica &= ~arches
    raw_solid = building | basilica
    free, records = _cleanup(con, inside, raw_solid)
    free, post_records = _post_cleanup(con, inside, free, basilica)
    building = inside & ~free & ~basilica  # buildings plus every filled cell
    records["arches"] = arch_records
    records.update(post_records)
    records["footprints_used"] = len(used)
    records["basilica_osm"] = basilica_osm
    records["raw_building_m2"] = round(float(raw_solid.sum()) * UNIT * UNIT, 2)
    return Layout(con, inside, free, building, basilica, records), feet, proj


def _line_cells(con: Construction, line: Sequence[Point]) -> BoolArray:
    """Cells crossed by a polyline, sampled every 0.05 m."""
    mask = np.zeros((con.ny, con.nx), dtype=bool)
    for (ax, ay), (bx, by) in zip(line, line[1:], strict=False):
        n = max(1, int(math.ceil(math.hypot(bx - ax, by - ay) / (UNIT / 2))))
        for k in range(n + 1):
            r, c = con.cell(ax + (bx - ax) * k / n, ay + (by - ay) * k / n)
            if 0 <= r < con.ny and 0 <= c < con.nx:
                mask[r, c] = True
    return mask


def _arches(
    osm: dict[str, Any], proj: Projection, con: Construction, inside: BoolArray, solid: BoolArray
) -> tuple[BoolArray, list[dict[str, Any]]]:
    """Decision 42: the cells within 1.0 m of every `tunnel=building_passage` highway inside the border."""
    carve = np.zeros_like(inside)
    records: list[dict[str, Any]] = []
    for e in osm["elements"]:
        tags = e.get("tags", {})
        if e["type"] != "way" or tags.get("tunnel") != "building_passage" or "highway" not in tags:
            continue
        line = [proj.xy(g["lat"], g["lon"]) for g in e["geometry"]]
        cells = _line_cells(con, line) & inside
        if not cells.any():
            continue
        near = (ndimage.distance_transform_edt(~cells) * UNIT <= ARCH_HALF_M + 1e-9) & inside
        records.append(
            {
                "osm": f"way/{e['id']}",
                "name": str(tags.get("name", "")),
                "length_m": round(_length(line), 1),
                "opened_m2": round(float((near & solid).sum()) * UNIT * UNIT, 2),
            }
        )
        carve |= near
    return carve, records


def _depinch(free: BoolArray, inside: BoolArray) -> tuple[BoolArray, int]:
    """Fill free cells where solid meets solid only diagonally (a 2 × 2 checkerboard), so every solid
    region's outline is a simple polygon; returns the free cells and how many were filled. Every free cell
    must lie inside the border: each pass then fills at least one cell, so the loop ends."""
    if bool((free & ~inside).any()):
        raise ValueError("RP3: free cells outside the border before the diagonal-contact fill")
    free = free.copy()
    filled = 0
    while True:
        a, b = free[:-1, :-1], free[:-1, 1:]
        c, d = free[1:, :-1], free[1:, 1:]
        hits = np.argwhere((a & d & ~b & ~c) | (b & c & ~a & ~d))
        if len(hits) == 0:
            return free, filled
        for r, c0 in hits:
            cell = (int(r), int(c0)) if free[int(r), int(c0)] else (int(r), int(c0) + 1)
            if free[cell]:  # an earlier fill in this pass may have resolved this contact
                free[cell] = False
                filled += 1


def _street(osm: dict[str, Any], proj: Projection, name: str) -> list[Point]:
    """The named street's centreline: its highway ways chained end to end by shared nodes."""
    ways = [
        e
        for e in osm["elements"]
        if e["type"] == "way"
        and e.get("tags", {}).get("name") == name
        and "highway" in e["tags"]
        and e["tags"].get("area") != "yes"
        and e["tags"]["highway"] not in ("footway", "steps")
    ]
    ends: dict[int, list[int]] = {}
    for i, w in enumerate(ways):
        for n in (w["nodes"][0], w["nodes"][-1]):
            ends.setdefault(n, []).append(i)
    starts = sorted(n for n, ws in ends.items() if len(ws) == 1)
    node = starts[0]
    seen: set[int] = set()
    chain: list[tuple[float, float]] = []
    while True:
        nxt = [i for i in ends.get(node, []) if i not in seen]
        if not nxt:
            break
        i = nxt[0]
        seen.add(i)
        w = ways[i]
        pts = [(g["lat"], g["lon"]) for g in w["geometry"]]
        if w["nodes"][0] != node:
            pts.reverse()
        chain += pts if not chain else pts[1:]
        node = w["nodes"][-1] if w["nodes"][0] == node else w["nodes"][0]
    if len(seen) != len(ways):
        raise ValueError(f"RP3: {name} is not one chain of ways")
    return [proj.xy(*p) for p in chain]


def _inside_part(line: list[Point], border: Sequence[Point], step: float = 0.05) -> list[Point]:
    """The longest run of the polyline (sampled every `step` m) that lies inside the border."""
    samples: list[Point] = []
    for (ax, ay), (bx, by) in zip(line, line[1:], strict=False):
        n = max(1, int(math.ceil(math.hypot(bx - ax, by - ay) / step)))
        samples += [(ax + (bx - ax) * k / n, ay + (by - ay) * k / n) for k in range(n)]
    samples.append(line[-1])
    inside = MplPath(np.asarray(border)).contains_points(np.asarray(samples))
    best: list[Point] = []
    run: list[Point] = []
    for pt, ok in zip(samples, inside, strict=True):
        if ok:
            run.append(pt)
        else:
            best, run = (run if len(run) > len(best) else best), []
    return run if len(run) > len(best) else best


def _at_length(line: list[Point], s: float) -> tuple[Point, float]:
    """The point at arc length s along the polyline and the local direction (radians)."""
    for (ax, ay), (bx, by) in zip(line, line[1:], strict=False):
        seg = math.hypot(bx - ax, by - ay)
        if s <= seg or (bx, by) == line[-1]:
            t = 0.0 if seg == 0 else min(1.0, s / seg)
            return (ax + (bx - ax) * t, ay + (by - ay) * t), math.atan2(by - ay, bx - ax)
        s -= seg
    return line[-1], 0.0


def _length(line: list[Point]) -> float:
    return sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(line, line[1:], strict=False))


def _snap(
    lay: Layout, clear: FloatArray, point: Point, street: list[Point] | None
) -> tuple[Point, float, float]:
    """The rule point if its cell has clearance ≥ 0.5 m, else the nearest free cell on the same street (within
    STREET_BAND_M of its centreline) with clearance ≥ 0.7 m; returns (cell centre, shift, clearance)."""
    con = lay.con
    r, c = con.cell(*point)
    if lay.free[r, c] and clear[r, c] >= SPAWN_CLEAR_M:
        x, y = con.centre(r, c)
        return (
            (round(x, 2), round(y, 2)),
            round(math.hypot(x - point[0], y - point[1]), 2),
            clear[r, c],
        )
    ok = lay.free & (clear >= SPAWN_FALLBACK_CLEAR_M)
    if street is not None:
        line = _line_cells(con, street)
        ok &= ndimage.distance_transform_edt(~line) * UNIT <= STREET_BAND_M
    cells = np.argwhere(ok)
    centres = (con.c0 + cells[:, 1] + 0.5) * UNIT, (con.r0 + cells[:, 0] + 0.5) * UNIT
    d2 = (centres[0] - point[0]) ** 2 + (centres[1] - point[1]) ** 2
    k = int(np.argmin(d2))
    r, c = int(cells[k, 0]), int(cells[k, 1])
    x, y = con.centre(r, c)
    return (round(x, 2), round(y, 2)), round(math.sqrt(float(d2[k])), 2), clear[r, c]


def _axis_near(direction: float) -> tuple[int, int]:
    """The axis rays (k of k·π/8) nearest to the two senses of a street direction."""
    ks = []
    for d in (direction, direction + math.pi):
        quarter = round(d / (math.pi / 2)) % 4
        ks.append(int(quarter) * 4)
    return ks[0], ks[1]


def _spawn_s1(
    lay: Layout,
    clear: FloatArray,
    occ: BoolArray,
    osm: dict[str, Any],
    proj: Projection,
    border: list[Point],
) -> tuple[Spawn, dict[str, Any]]:
    """S1: Carrer de Montcada, midpoint of its part inside the border."""
    montcada = _inside_part(_street(osm, proj, "Carrer de Montcada"), border)
    mid, direction = _at_length(montcada, _length(montcada) / 2)
    s1, s1_shift, s1_clear = _snap(lay, clear, mid, montcada)
    k1, k2 = _axis_near(direction)
    s1_yaw = longest_ray_heading(lay.con.grid(), occ, s1[0], s1[1], (k1, k2))
    rec = {
        "rule_point": [round(mid[0], 2), round(mid[1], 2)],
        "inside_length_m": round(_length(montcada), 1),
        "street_bearing_deg": round(math.degrees(direction), 1),
        "shift_m": s1_shift,
        "candidate_rays": [k1, k2],
        "clearance_m": round(float(s1_clear), 2),
    }
    spawn = Spawn(
        "S1",
        s1[0],
        s1[1],
        s1_yaw,
        "Carrer de Montcada, midpoint of its part inside the border; along the street "
        "(the longer of the two axis rays nearest its direction)",
    )
    return spawn, rec


def _spawn_s2(
    lay: Layout,
    clear: FloatArray,
    occ: BoolArray,
    osm: dict[str, Any],
    proj: Projection,
    border: list[Point],
) -> tuple[Spawn, dict[str, Any]]:
    """S2: Carrer de l'Argenteria, 20 m inside the west end of its part inside the border."""
    arg = _inside_part(_street(osm, proj, "Carrer de l'Argenteria"), border)
    if arg[0][0] > arg[-1][0]:
        arg = arg[::-1]
    p2, _ = _at_length(arg, S2_INSET_M)
    s2, s2_shift, s2_clear = _snap(lay, clear, p2, arg)
    s2_yaw = longest_ray_heading(lay.con.grid(), occ, s2[0], s2[1], AXIS_RAYS)
    rec = {
        "rule_point": [round(p2[0], 2), round(p2[1], 2)],
        "inside_length_m": round(_length(arg), 1),
        "shift_m": s2_shift,
        "clearance_m": round(float(s2_clear), 2),
    }
    spawn = Spawn(
        "S2",
        s2[0],
        s2[1],
        s2_yaw,
        "Carrer de l'Argenteria, 20 m inside the west end of its part inside the border; "
        "longest axis ray",
    )
    return spawn, rec


def _spawn_s3(lay: Layout, clear: FloatArray, occ: BoolArray) -> tuple[Spawn, dict[str, Any]]:
    """S3: the widest free cell within 30 m of the basilica's west front."""
    con = lay.con
    rows, cols = np.nonzero(lay.basilica)
    bx = (con.c0 + cols + 0.5) * UNIT
    by = (con.r0 + rows + 0.5) * UNIT
    pts = np.column_stack([bx, by])
    centre = pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts - centre, full_matrices=False)
    axis = vt[0] if vt[0][0] < 0 else -vt[0]  # the long axis, pointing to the western end
    front = pts[int(np.argmax((pts - centre) @ axis))]
    xs, ys = con.centres()
    gx, gy = np.meshgrid(xs, ys)
    near = lay.free & ((gx - front[0]) ** 2 + (gy - front[1]) ** 2 <= S3_RADIUS_M**2)
    masked = np.where(near, clear, -1.0)
    r3, c3 = (int(v) for v in np.unravel_index(int(np.argmax(masked)), masked.shape))
    x3, y3 = con.centre(r3, c3)
    s3 = (round(x3, 2), round(y3, 2))
    s3_yaw = longest_ray_heading(con.grid(), occ, s3[0], s3[1], AXIS_RAYS)
    rec = {
        "west_front": [round(float(front[0]), 2), round(float(front[1]), 2)],
        "clearance_m": round(float(clear[r3, c3]), 2),
    }
    spawn = Spawn(
        "S3",
        s3[0],
        s3[1],
        s3_yaw,
        "widest free cell within 30 m of the basilica's west front; longest axis ray",
    )
    return spawn, rec


def _spawns(
    lay: Layout, osm: dict[str, Any], proj: Projection, border: list[Point]
) -> tuple[tuple[Spawn, ...], dict[str, Any]]:
    occ = ~lay.free
    clear: FloatArray = ndimage.distance_transform_edt(lay.free) * UNIT - UNIT / 2
    s1, rec1 = _spawn_s1(lay, clear, occ, osm, proj, border)
    s2, rec2 = _spawn_s2(lay, clear, occ, osm, proj, border)
    s3, rec3 = _spawn_s3(lay, clear, occ)
    return (s1, s2, s3), {"S1": rec1, "S2": rec2, "S3": rec3}


def _simple(poly: Sequence[Point]) -> bool:
    """No two non-adjacent edges of the closed polygon intersect."""
    n = len(poly)

    def cross(o: Point, a: Point, b: Point) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i or (j + 1) % n == i or j == (i + 1) % n:
                continue
            c, d = poly[j], poly[(j + 1) % n]
            d1, d2 = cross(c, d, a), cross(c, d, b)
            d3, d4 = cross(a, b, c), cross(a, b, d)
            if d1 * d2 < 0 and d3 * d4 < 0:
                return False
    return True


def _make_checks(lay: Layout, border: dict[str, Any]) -> ExtraCheck:
    poly = [(float(x), float(y)) for x, y in border["border_m"]]
    residual = float(border["residual_max_m"])

    def checks(world: World, a: Analysis) -> list[CheckResult]:
        results = [no_symmetry("RP3 layout has no symmetry", world)]
        # the ring is stored open (closed implicitly), so its last vertex must not repeat the first
        no_repeated_vertex = len(poly) >= 3 and poly[0] != poly[-1]
        results.append(
            CheckResult(
                "RP3 border polygon closed and simple; georeferencing residual ≤ 3 m",
                no_repeated_vertex and _simple(poly) and residual <= RESIDUAL_MAX_M,
                {"vertices": len(poly), "simple": _simple(poly), "residual_max_m": residual},
            )
        )
        con = lay.con
        grid = con.grid()
        statics = world.static_primitives()
        solids = tuple(p for p in statics if p.role is not Role.BOUNDARY_WALL)
        walls = tuple(p for p in statics if p.role is Role.BOUNDARY_WALL)
        solid = raster.rasterise(grid, solids, slab=None)
        wall = raster.rasterise(grid, walls, slab=None)
        reach = ndimage.distance_transform_edt(~lay.inside) * UNIT
        beyond = solid & ~lay.inside
        reach_max = round(float(reach[beyond].max()), 2) if beyond.any() else 0.0
        wall_inside = int(np.count_nonzero(wall & lay.inside))
        free_outside = int(np.count_nonzero(a.reachable & ~a.inclusion))
        # a solid reaches beyond the border by its EXT_CELLS extension (0.3 m), plus at most the coarsest
        # simplification tolerance (0.12 m), plus one construction cell (0.1 m) of rasterisation: 0.52 m
        limit = EXT_CELLS * UNIT + max(DP_EPS_M) + UNIT
        results.append(
            CheckResult(
                "RP3 solids inside the border except where they reach under the boundary wall "
                f"(≤ {limit:.2f} m beyond the border); boundary wall outside the border; no free cell "
                "outside it",
                reach_max <= limit + 1e-9 and wall_inside == 0 and free_outside == 0,
                {
                    "solid_beyond_border_max_m": reach_max,
                    "wall_cells_inside": wall_inside,
                    "free_cells_outside": free_outside,
                },
            )
        )
        bad = [p.name for p in solids if polygon_problem(p.vertices)]
        results.append(
            CheckResult(
                "RP3 solids are simple counter-clockwise polygons",
                not bad,
                {"polygons": len(solids), "invalid": bad[:5]},
            )
        )
        labels, n = ndimage.label(a.reachable | a.sealed)
        results.append(
            CheckResult(
                "RP3 street network one connected component after clean-up",
                n == 1 and not a.sealed.any(),
                {"components": int(n)},
            )
        )
        narrow = a.narrowest_m
        results.append(
            CheckResult(
                "RP3 clean-up recorded; no passage below 1.0 m (one 0.05 m cell tolerance)",
                narrow is not None and narrow >= 1.0 - a.grid.h - 1e-9,
                {
                    "narrowest_m": narrow,
                    "alleys_filled": lay.records["step3_narrow"]["alleys"],
                },
            )
        )
        return results

    return checks


def _dp_ring(ring: Sequence[Point], eps: float) -> list[Point]:
    """Douglas–Peucker on a closed ring (iterative), split at vertex 0 and the vertex farthest from it."""
    pts = np.asarray(ring, dtype=np.float64)
    far = int(np.argmax(np.hypot(*(pts - pts[0]).T)))
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = keep[far] = True
    stack = [(0, far), (far, len(pts))]
    while stack:
        i0, i1 = stack.pop()
        a = pts[i0]
        b = pts[i1 % len(pts)]
        if i1 - i0 < 2:
            continue
        seg = pts[i0 + 1 : i1]
        ab = b - a
        length = float(np.hypot(*ab))
        if length > 0:
            dist = np.abs(ab[0] * (seg[:, 1] - a[1]) - ab[1] * (seg[:, 0] - a[0])) / length
        else:
            dist = np.hypot(*(seg - a).T)
        k = int(np.argmax(dist))
        if dist[k] > eps:
            mid = i0 + 1 + k
            keep[mid] = True
            stack += [(i0, mid), (mid, i1)]
    return [(float(x), float(y)) for x, y in pts[keep]]


def _outline(con: Construction, mask: BoolArray) -> tuple[list[Point], float]:
    """The simplified outline of one solid region: its traced cell-edge contour, Douglas–Peucker with the
    first tolerance in DP_EPS_M that gives a valid polygon (vertices rounded to 1 mm)."""
    rows, cols = np.nonzero(mask)
    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1
    x0, y0 = con.centre(r0, c0)
    sub = raster.Grid(round(x0 - UNIT / 2, 6), round(y0 - UNIT / 2, 6), c1 - c0, r1 - r0, UNIT)
    ring = raster.trace_outline(sub, mask[r0:r1, c0:c1])
    for eps in DP_EPS_M:
        poly = [(round(x, 3), round(y, 3)) for x, y in _dp_ring(ring, eps)]
        if not polygon_problem(poly):
            return poly, eps
    poly = [(round(x, 3), round(y, 3)) for x, y in ring]
    problem = polygon_problem(poly)
    if problem:
        raise ValueError(f"RP3: a solid region's outline is invalid: {problem}")
    return poly, 0.0


def _solids(lay: Layout) -> tuple[list[Primitive], dict[str, Any]]:
    """Decision 41: one extruded polygon per 4-connected solid region (buildings, clean-up fill) and one for the
    basilica. Each region reaches EXT_CELLS cells under the boundary wall and into a touching region of the
    other kind, so simplification never opens a sliver between them; regions have no holes (courtyards were
    filled), which is verified."""
    con = lay.con
    beyond = ~lay.inside & (ndimage.distance_transform_edt(lay.inside == 0) <= EXT_CELLS + 1e-9)
    disk = _disk(EXT_CELLS)
    prims: list[Primitive] = []
    eps_used: dict[str, int] = {}
    for kind, own, other in (
        ("bldg", lay.building, lay.basilica),
        ("basilica", lay.basilica, lay.building),
    ):
        grown = own | (ndimage.binary_dilation(own, structure=disk) & (other | beyond))
        labels, n = _components(grown)
        for i, sl in enumerate(ndimage.find_objects(labels), start=1):
            if sl is None:
                continue
            region = np.zeros_like(own)
            region[sl] = labels[sl] == i
            if not (region & own).any():
                continue
            filled = ndimage.binary_fill_holes(region[sl])
            if bool((filled & ~region[sl]).any()):
                raise ValueError(f"RP3: solid region {kind} {i} has a hole")
            poly, eps = _outline(con, region)
            eps_used[f"{eps:g}"] = eps_used.get(f"{eps:g}", 0) + 1
            if kind == "bldg":
                prims.append(prism(f"bldg_{len(prims):03d}", poly, BUILDING_H, Role.WALL, "inner"))
            else:
                prims.append(
                    prism(f"basilica_{i:02d}", poly, BASILICA_H, Role.LANDMARK, "landmark")
                )
    return prims, {"dp_tolerance_counts": dict(sorted(eps_used.items()))}


def _boundary_wall(poly: Sequence[Point]) -> list[Primitive]:
    """One 0.2 m box per border edge, its inner face on the edge and outside the border (the polygon is
    counter-clockwise, so outside is to the right); at convex corners each box runs past the corner by
    WALL_T·tan(turn/2) + 1 cm so the outer faces close."""
    n = len(poly)
    prims: list[Primitive] = []
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        dx, dy = b[0] - a[0], b[1] - a[1]
        length = math.hypot(dx, dy)
        ux, uy = dx / length, dy / length

        def turn(p: Point, q: Point, r: Point) -> float:
            u = math.atan2(q[1] - p[1], q[0] - p[0])
            v = math.atan2(r[1] - q[1], r[0] - q[0])
            return (v - u + math.pi) % (2 * math.pi) - math.pi  # > 0: left turn (convex)

        t0 = turn(poly[i - 1], a, b)
        t1 = turn(a, b, poly[(i + 2) % n])
        e0 = WALL_T * math.tan(t0 / 2) + 0.01 if t0 > 0 else 0.0
        e1 = WALL_T * math.tan(t1 / 2) + 0.01 if t1 > 0 else 0.0
        mx = (a[0] + b[0]) / 2 + ux * (e1 - e0) / 2 + uy * WALL_T / 2
        my = (a[1] + b[1]) / 2 + uy * (e1 - e0) / 2 - ux * WALL_T / 2
        prims.append(
            Primitive(
                Kind.BOX,
                f"boundary_{i:03d}",
                round(mx, 4),
                round(my, 4),
                round(math.atan2(dy, dx), 6),
                (round(length + e0 + e1, 4), WALL_T, WALL_H),
                Role.BOUNDARY_WALL,
                "boundary",
            )
        )
    return prims


def build() -> World:
    osm = _read(OSM_FILE, OSM_SHA256)
    border = _read(BORDER_FILE, BORDER_SHA256)
    lay, feet, proj = _layout(osm, border)
    poly = tuple((float(x), float(y)) for x, y in border["border_m"])
    solids, solid_rec = _solids(lay)
    prims = (*_boundary_wall(poly), *solids)
    # the free space the primitives leave on the construction grid (cell centres), its main component
    occ = raster.rasterise(lay.con.grid(), prims, slab=None)
    free, _ = _largest_component(lay.inside & ~occ)
    final = replace(lay, free=free)
    spawns, spawn_rec = _spawns(final, osm, proj, list(poly))
    area = round(float(np.count_nonzero(final.free)) * UNIT * UNIT, 2)
    border_area = round(float(np.count_nonzero(lay.inside)) * UNIT * UNIT, 1)
    roles = {r.value: sum(1 for p in prims if p.role is r) for r in Role}
    return World(
        id="RP3",
        slug="santa_maria_del_mar",
        title="Santa Maria del Mar, El Born (Barcelona)",
        tier="RP",
        role="evaluation",
        plan="docs/plans/worlds/R10-santa-maria-del-mar.md",
        primitives=prims,
        spawns=spawns,
        expected=Expected(
            free_area_m2=area,
            narrowest_passage_m=1.0,
            narrowest_tolerance_m=0.5,
        ),
        inclusion=(poly,),
        sources=(
            {
                "what": "building footprints, streets and names (Overpass extract, sources/RP3/)",
                "cite": "OpenStreetMap contributors",
            },
            {
                "what": "border drawn by the maintainer on the OpenStreetMap map, georeferenced "
                "(sources/RP3/border.json)",
                "cite": "https://www.openstreetmap.org/?#map=18/41.384370/2.180633",
            },
        ),
        extra_checks=(_make_checks(lay, border),),
        plan_data={
            "border": [[round(x, 2), round(y, 2)] for x, y in poly],
            "border_area_m2": float(border["area_m2"]),
            "border_area_construction_m2": border_area,
            "projection": {
                **border["projection"],
                "construction_grid_m": UNIT,
                "construction_origin": list(lay.con.origin),
                "grid": [lay.con.nx, lay.con.ny],
            },
            "georeferencing": {
                "control_points": len(border["control_points"]),
                "residual_rms_m": border["residual_rms_m"],
                "residual_max_m": border["residual_max_m"],
                "check_against_building_registration_m": border[
                    "check_against_building_registration_m"
                ],
            },
            "source_sha256": {OSM_FILE: OSM_SHA256, BORDER_FILE: BORDER_SHA256},
            "footprints_in_extract": len(feet),
            "cleanup": {k: v for k, v in lay.records.items() if k.startswith("step")},
            "footprints_used": lay.records["footprints_used"],
            "basilica_osm": lay.records["basilica_osm"],
            "raw_building_m2": lay.records["raw_building_m2"],
            "construction_free_area_m2": area,
            "primitives_by_role": {k: v for k, v in roles.items() if v},
            "solids": solid_rec,
            "arches": lay.records["arches"],
            "stub": lay.records["stub"],
            "diagonal_contacts_filled": lay.records["diagonal_contacts_filled"],
            "border_slivers_opened_m2": lay.records["border_slivers_opened_m2"],
            "princesa_edge": border["princesa_edge"],
            "border_drawn_area_m2": float(border["area_drawn_m2"]),
            "spawn_rules": spawn_rec,
        },
        plan_width_px=3200,
        plan_detail_m=60.0,
    )
