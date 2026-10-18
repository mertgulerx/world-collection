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

"""`world.json` for Blender, `world.yaml` and hashes (T0 §10)."""

from __future__ import annotations

import hashlib
import json
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

from . import WORLDGEN_VERSION, moving
from .checks import Analysis
from .sdf import FLOOR_MARGIN_M
from .world import (
    MATERIALS,
    CheckResult,
    Kind,
    Notch,
    PlanKey,
    Primitive,
    Spawn,
    World,
    sorted_primitives,
)

STATUS_KEYS = ("status", "review")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _r(value: float) -> float:
    return round(float(value), 4)


def _primitive_json(p: Primitive) -> dict[str, object]:
    data: dict[str, object] = {
        "name": p.name,
        "kind": p.kind.value,
        "x": _r(p.x),
        "y": _r(p.y),
        "z_min": _r(p.z_min),
        "yaw": round(p.yaw, 6),
        "size": [_r(v) for v in p.size],
        "role": p.role.value,
        "material": p.material,
    }
    if (
        p.kind is Kind.POLYGON
    ):  # only polygon prisms carry vertices, so older world.json stay identical
        data["vertices"] = [[_r(x), _r(y)] for x, y in p.vertices]
    return data


def _spawn(s: Spawn) -> dict[str, object]:
    """A spawn as `world.json` and `world.yaml` write it (the note is not exported)."""
    return {"id": s.id, "x": _r(s.x), "y": _r(s.y), "yaw": round(s.yaw, 6)}


def _notch(n: Notch) -> dict[str, object]:
    return {"id": n.id, "polygon": [[_r(x), _r(y)] for x, y in n.polygon]}


def world_json(world: World) -> str:
    """The scene description the Blender script reads: primitives, floor, spawns and notches."""
    xmin, xmax, ymin, ymax = world.bounds()
    data = {
        "id": world.id,
        "title": world.title,
        "bounds": [_r(xmin), _r(xmax), _r(ymin), _r(ymax)],
        "floor_margin_m": FLOOR_MARGIN_M,
        "materials": {k: list(v) for k, v in MATERIALS.items()},
        "primitives": [
            _primitive_json(p)
            for p in sorted_primitives(world.primitives)
            + (moving.primitives_at(world.moving) if world.moving is not None else ())
        ],
        "spawns": [_spawn(s) for s in world.spawns],
        "notches": [_notch(n) for n in world.notches],
    }
    if (
        world.overview_elevation_deg != 45.0
    ):  # written only when it differs, so older world.json stay identical
        data["overview_elevation_deg"] = world.overview_elevation_deg
    return json.dumps(data, indent=1, sort_keys=False) + "\n"


def _version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def world_yaml_data(
    world: World,
    a: Analysis,
    sdf_text: str,
    json_text: str,
    moving_measured: moving.Measured | None = None,
) -> dict[str, Any]:
    """The `world.yaml` data; the `moving_objects` block (E0 §6) only for worlds with moving objects."""
    exp = world.expected
    expected: dict[str, Any] = {
        "free_area_m2": exp.free_area_m2,
        "free_area_tolerance": exp.free_area_tolerance,
        "narrowest_passage_m": exp.narrowest_passage_m,
        "minimum_passage_m": exp.minimum_passage_m,
    }
    if exp.narrowest_tolerance_m is not None:
        expected["narrowest_tolerance_m"] = exp.narrowest_tolerance_m
    if exp.loops is not None:
        expected["loops"] = exp.loops
    if exp.loops_unsized is not None:
        expected["loops_unsized"] = exp.loops_unsized
    return {
        "id": world.id,
        "slug": world.slug,
        "title": world.title,
        "tier": world.tier,
        "role": world.role,
        "family": world.family,
        "worldgen_version": WORLDGEN_VERSION,
        "generator": {
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "matplotlib": _version("matplotlib"),
        },
        "plan": world.plan,
        "parameters": dict(sorted(world.parameters.items())),
        "seed": world.seed,
        "selection": world.plan_data.get(PlanKey.SELECTION),
        "expected": expected,
        "measured": {
            "free_area_m2": _r(a.free_area_m2),
            "narrowest_passage_m": a.narrowest_m,
            "loops": a.loops,
            "loops_unsized": a.loops_unsized,
        },
        "descriptors": None,
        "spawns": [_spawn(s) for s in world.spawns],
        "inclusion": [[[_r(x), _r(y)] for x, y in poly] for poly in a.inclusion_polygons],
        "notches": [_notch(n) for n in world.notches],
        "doors": [
            {
                "id": d.id,
                "centre": [_r(d.x), _r(d.y)],
                "width_m": _r(d.width),
                "wall_axis": d.wall_axis,
            }
            for d in world.doors
        ],
        "moving_objects": (
            None if world.moving is None else moving.yaml_block(world.moving, moving_measured)
        ),
        "plan_data": {k: v for k, v in sorted(world.plan_data.items()) if k != PlanKey.SELECTION},
        "sources": [dict(s) for s in world.sources],
        "licence": world.licence,
        "hashes": {"sdf": sha256_text(sdf_text), "world_json": sha256_text(json_text)},
        "status": "built",
        "review": {"image_review": None, "maintainer": None},
    }


def dump_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(
        data, sort_keys=False, allow_unicode=True, width=120, default_flow_style=None
    )


def keep_status(data: dict[str, Any], previous: Path) -> dict[str, Any]:
    """Keep the review status of an existing world.yaml when the SDF is unchanged."""
    if not previous.exists():
        return data
    old = yaml.safe_load(previous.read_text(encoding="utf-8")) or {}
    if old.get("hashes", {}).get("sdf") == data["hashes"]["sdf"]:
        for key in STATUS_KEYS:
            if key in old:
                data[key] = old[key]
    return data


def checks_json(results: list[CheckResult]) -> str:
    rows = [
        {"check": r.name, "passed": r.passed, "value": r.value, "detail": r.detail} for r in results
    ]
    return (
        json.dumps(
            {"all_passed": all(r.passed for r in results), "checks": rows}, indent=1, default=str
        )
        + "\n"
    )
