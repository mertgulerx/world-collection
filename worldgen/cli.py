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

"""Command line (T0 §9): build, emit, emit-moving-test, review, review-family, status.

Run from `world_collection/` with the `.venv-suite` interpreter: `python -m worldgen build B1`.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from . import catalog, checks, descriptors, export, moving, moving_review, plan_plot
from .families.common import SelectionFailed
from .sdf import write_sdf
from .world import CheckResult, PlanKey, World

ROOT = Path(__file__).resolve().parents[1]
WORLDS_DIR = ROOT / "worlds"
REPO_LICENSE = ROOT.parent / "LICENSE"  # worlds carry the repository licence, Apache-2.0
BLENDER = shutil.which("blender") or "blender"
BLENDER_TIMEOUT_S = 3600.0
EMIT_TIMEOUT_S = 600.0


def moving_measured(world: World, analysis: checks.Analysis) -> moving.Measured | None:
    """M1-M6 values of a world's moving objects over the check horizon (E0 §8); None for static worlds."""
    if world.moving is None:
        return None
    ctx = moving.check_context(world, analysis, world.moving.doorways, world.moving.limits)
    return moving.measure(world.moving.objects, ctx)


@dataclass(frozen=True, slots=True)
class Emitted:
    """What `emit` wrote and computed."""

    sdf_text: str
    yaml_text: str  # the fresh world.yaml text, review status not kept; `emit` does not write it
    analysis: checks.Analysis
    data: dict[str, Any]  # the world.yaml data behind `yaml_text`
    measured: moving.Measured | None  # moving-object measures; None for static worlds


def emit(world: World, folder: Path) -> Emitted:
    """Write the SDF, `review/world.json`, LICENSE and NOTICE into `folder`, compute the analysis, the
    moving-object measures and the descriptors, and return them with the fresh `world.yaml` text. The caller
    writes `world.yaml`: `build` keeps the previous review status, `emit` does not."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "review").mkdir(exist_ok=True)
    sdf_text = write_sdf(world)
    json_text = export.world_json(world)
    analysis = checks.analyse(world)
    measured = moving_measured(world, analysis)
    data = export.world_yaml_data(world, analysis, sdf_text, json_text, measured)
    data["descriptors"] = descriptors.compute(world, analysis)
    (folder / f"{world.folder}.sdf").write_text(sdf_text, encoding="utf-8")
    (folder / "review" / "world.json").write_text(json_text, encoding="utf-8")
    if world.licence == "Apache-2.0":
        (folder / "LICENSE").write_text(REPO_LICENSE.read_text(encoding="utf-8"), encoding="utf-8")
    if world.notice:
        (folder / "NOTICE").write_text(world.notice, encoding="utf-8")
    return Emitted(sdf_text, export.dump_yaml(data), analysis, data, measured)


def print_results(results: Iterable[CheckResult]) -> None:
    for r in results:
        print(f"{'PASS' if r.passed else 'FAIL'}  {r.name}: {r.value} {r.detail}".rstrip())


def moving_review_pngs(world: World, analysis: checks.Analysis, review: Path) -> None:
    """`paths.png` and `timelapse.png` of a world with moving objects (T0 §8)."""
    moving_review.paths_png(world, analysis, review / "paths.png")
    moving_review.timelapse_png(world, analysis, review / "timelapse.png")


def determinism(world: World, sdf_text: str, yaml_text: str) -> CheckResult:
    """Check 8: a second build in a fresh interpreter and a temporary folder gives identical files."""
    with tempfile.TemporaryDirectory(prefix="worldgen_") as tmp:
        cmd = [sys.executable, "-m", "worldgen", "emit", world.id, "--out", tmp]
        run = subprocess.run(
            cmd, cwd=ROOT, capture_output=True, text=True, timeout=EMIT_TIMEOUT_S, check=False
        )
        if run.returncode != 0:
            return CheckResult("8 determinism", False, None, run.stderr[-400:])
        sdf2 = (Path(tmp) / f"{world.folder}.sdf").read_text(encoding="utf-8")
        yaml2 = (Path(tmp) / "world.yaml").read_text(encoding="utf-8")
    same = sdf2 == sdf_text and yaml2 == yaml_text
    return CheckResult(
        "8 determinism", same, export.sha256_text(sdf_text)[:16], "" if same else "outputs differ"
    )


def cmd_emit(args: argparse.Namespace) -> int:
    world = catalog.get(args.id)
    out = emit(world, Path(args.out))
    (Path(args.out) / "world.yaml").write_text(out.yaml_text, encoding="utf-8")
    return 0


def cmd_build(args: argparse.Namespace) -> int:
    world = catalog.get(args.id)
    folder = WORLDS_DIR / world.folder
    started = time.monotonic()
    out = emit(world, folder)
    analysis = out.analysis
    results = checks.run_checks(world, analysis, folder / f"{world.folder}.sdf")
    if world.moving is not None and out.measured is not None:
        block = out.data["moving_objects"]
        assert isinstance(block, dict)
        results.extend(moving.run_checks(world, analysis, out.sdf_text, block, out.measured))
    results.append(determinism(world, out.sdf_text, out.yaml_text))
    data = export.keep_status(out.data, folder / "world.yaml")
    (folder / "world.yaml").write_text(export.dump_yaml(data), encoding="utf-8")
    (folder / "review" / "checks.json").write_text(export.checks_json(results), encoding="utf-8")
    (folder / "review" / "descriptors.json").write_text(
        json.dumps(data["descriptors"], indent=1) + "\n", encoding="utf-8"
    )
    print_results(results)
    ok = all(r.passed for r in results)
    print(
        f"{world.id}: {'all checks passed' if ok else 'CHECKS FAILED'} ({time.monotonic() - started:.1f} s)"
    )
    if ok and not args.no_plan:
        plan_plot.plan_png(world, analysis, folder / "review" / "plan.png", world.plan_width_px)
        if world.plan_detail_m is not None:
            s, half = world.spawns[0], world.plan_detail_m / 2
            crop = (s.x - half, s.x + half, s.y - half, s.y + half)
            plan_plot.plan_png(world, analysis, folder / "review" / "plan_detail.png", 1600, crop)
        plan_plot.skeleton_png(world, analysis, folder / "review" / "skeleton.png")
    if ok and world.moving is not None:
        moving_review_pngs(world, analysis, folder / "review")
    return 0 if ok else 1


def cmd_review(args: argparse.Namespace) -> int:
    world = catalog.get(args.id)
    folder = WORLDS_DIR / world.folder
    report = folder / "review" / "checks.json"
    if not report.exists() or not json.loads(report.read_text(encoding="utf-8"))["all_passed"]:
        print(
            f"{world.id}: build it first; renders run only on a world whose checks passed",
            file=sys.stderr,
        )
        return 1
    script = Path(__file__).with_name("blender_scene.py")
    cmd = [
        BLENDER,
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "1",
        "--python",
        str(script),
        "--",
        "--world",
        str(folder / "review" / "world.json"),
        "--out",
        str(folder / "review"),
        "--only",
        args.only,
    ]
    started = time.monotonic()
    run = subprocess.run(
        cmd, capture_output=True, text=True, timeout=BLENDER_TIMEOUT_S, check=False
    )
    rendered = [line for line in run.stdout.splitlines() if line.startswith("RENDERED")]
    print("\n".join(rendered))
    if run.returncode != 0:
        print(run.stdout[-2000:], run.stderr[-2000:], file=sys.stderr)
        return run.returncode
    print(f"{world.id}: {len(rendered)} renders ({time.monotonic() - started:.0f} s)")
    return 0


def cmd_review_family(args: argparse.Namespace) -> int:
    """`review-family <Cn>`: the family contact sheet and `family.yaml` (T0 §8, §9)."""
    info = catalog.FAMILIES.get(args.family)
    if info is None:
        print(
            f"unknown family {args.family!r}; known: {', '.join(catalog.FAMILIES)}", file=sys.stderr
        )
        return 1
    entries: list[tuple[World, checks.Analysis, dict[str, Any]]] = []
    for world_id in info.instances:
        world = catalog.get(world_id)
        path = WORLDS_DIR / world.folder / "world.yaml"
        if not path.exists():
            print(f"{world_id}: build it first", file=sys.stderr)
            return 1
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        entries.append((world, checks.analyse(world), data))
    builds = {
        data["id"]: (data["worldgen_version"], json.dumps(data["generator"], sort_keys=True))
        for _, _, data in entries
    }
    if len(set(builds.values())) > 1:
        print(
            f"{args.family}: instances built by different generators; rebuild them:",
            file=sys.stderr,
        )
        for world_id, (version, generator) in builds.items():
            print(f"  {world_id}: worldgen {version}, {generator}", file=sys.stderr)
        return 1
    folder = WORLDS_DIR / f"{args.family}_family"
    (folder / "review").mkdir(parents=True, exist_ok=True)
    plan_plot.family_sheet(
        args.family, entries, info.columns, folder / "review" / "family_contact_sheet.png"
    )
    instances = []
    for world, _, data in entries:
        selection = data.get(PlanKey.SELECTION) or {}
        instances.append(
            {
                "id": data["id"],
                "folder": world.folder,
                "parameters": data["parameters"],
                "seed": data["seed"],
                "candidate": selection.get("candidate"),
                "rejected": len(selection.get("rejected", [])),
                "status": data["status"],
                "sdf_sha256": data["hashes"]["sdf"],
            }
        )
    family = {
        "family": args.family,
        "plan": info.plan,
        "worldgen_version": entries[0][2]["worldgen_version"],
        "generator": entries[0][2]["generator"],
        "seed_sequences": info.seeds,
        "rejected_total": sum(int(i["rejected"]) for i in instances),
        "instances": instances,
    }
    (folder / "family.yaml").write_text(export.dump_yaml(family), encoding="utf-8")
    print(f"{args.family}: contact sheet and family.yaml for {len(entries)} instances")
    return 0


def cmd_emit_moving_test(args: argparse.Namespace) -> int:
    """The E0 scratch world (plan §10 steps 3-4): SDF, world.yaml, review images and expected poses."""
    out = Path(args.out)
    world = moving.test_world(moving.VelocityMode(args.velocity_mode))
    world.validate()
    assert world.moving is not None
    emitted = emit(world, out)
    (out / "world.yaml").write_text(emitted.yaml_text, encoding="utf-8")
    if args.stamps_file:
        text = Path(args.stamps_file).read_text(encoding="utf-8").split()
        stamps = [int(v) for v in text]
    else:
        step = args.period_ms * 1_000_000
        stamps = list(range(0, round(args.duration_s * 1e9) + 1, step))
    poses = moving.expected_poses(world.moving.objects, stamps)
    (out / "expected_poses.json").write_text(json.dumps(poses) + "\n", encoding="utf-8")
    if not args.no_review:
        moving_review_pngs(world, emitted.analysis, out / "review")
    assert emitted.measured is not None
    block = emitted.data["moving_objects"]
    assert isinstance(block, dict)
    print_results(
        moving.run_checks(world, emitted.analysis, emitted.sdf_text, block, emitted.measured)
    )
    print(f"{out / f'{world.folder}.sdf'}; {len(stamps)} expected stamps in expected_poses.json")
    return 0


def cmd_status(_: argparse.Namespace) -> int:
    for path in sorted(WORLDS_DIR.glob("*/world.yaml")):
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
        print(f"{data['id']:8} {data['status']:10} {data['title']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="worldgen")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build", help="write the world folder and run the checks")
    p.add_argument("id")
    p.add_argument("--no-plan", action="store_true", help="skip plan.png")
    p.set_defaults(func=cmd_build)
    p = sub.add_parser(
        "emit", help="write SDF and world.yaml to a folder (used by the determinism check)"
    )
    p.add_argument("id")
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_emit)
    p = sub.add_parser(
        "emit-moving-test",
        help="E0 scratch world with three moving objects, plus expected poses (plan E0 §10)",
    )
    p.add_argument("--out", required=True)
    p.add_argument(
        "--velocity-mode",
        default=moving.VelocityMode.POSE_AND_VELOCITY.value,
        choices=[v.value for v in moving.VelocityMode],
    )
    p.add_argument(
        "--duration-s", type=float, default=600.0, help="expected poses from 0 to this time"
    )
    p.add_argument("--period-ms", type=int, default=50, help="spacing of the expected stamps")
    p.add_argument(
        "--stamps-file", default="", help="whitespace-separated nanosecond stamps instead"
    )
    p.add_argument("--no-review", action="store_true", help="skip paths.png and timelapse.png")
    p.set_defaults(func=cmd_emit_moving_test)
    p = sub.add_parser("review", help="Blender renders of a built world")
    p.add_argument("id")
    p.add_argument("--only", default="", help="comma-separated: overview, top, eye_S1, ...")
    p.set_defaults(func=cmd_review)
    p = sub.add_parser("review-family", help="family contact sheet and family.yaml")
    p.add_argument("family")
    p.set_defaults(func=cmd_review_family)
    p = sub.add_parser("status", help="status of every built world")
    p.set_defaults(func=cmd_status)
    args = parser.parse_args(argv)
    try:
        result: int = args.func(args)
    except SelectionFailed as exc:  # a generated family found no acceptable candidate
        raise SystemExit(str(exc)) from None
    return result
