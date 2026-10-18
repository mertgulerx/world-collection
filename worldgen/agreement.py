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


"""Python/C++ agreement of the moving-object system (E0 §10, step 4).

Compares a recording of the system's pose topic with the reference evaluator of `moving.py` at equal stamps:

    gz topic -e -t /benchmark/moving_objects/pose > recording.txt     # while the world runs
    python -m worldgen.agreement --recording recording.txt --world world.sdf

The recording is the text form `gz topic -e` prints for `gz.msgs.Pose_V`: one top-level `header` with the
simulation stamp, then one `pose` per object (`name`, `position`, `orientation`; zero fields are omitted). The
objects and their paths are read from the world's SDF plugin element, as the system reads them. For every
object the tool prints the number of samples and the maximum position deviation (m, in the plane) and yaw
deviation (rad, wrapped to (-pi, pi]) between the recorded pose and p(t) at the recorded stamp t. The exit status
is 1 when a deviation exceeds the pass limits (1 mm, 1 mrad), when an object has no sample, or when the recording
names an object the SDF does not have.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

from .moving import ParsedObject, objects_from_sdf, pose_at, wrap

POSITION_LIMIT_M = 1e-3
YAW_LIMIT_RAD = 1e-3

_HEADER = re.compile(r"(?m)^header \{")
_STAMP = re.compile(r"stamp \{\s*(?:sec: (\d+)\s*)?(?:nsec: (\d+)\s*)?\}")
_POSE = re.compile(r'name: "([^"]+)"\s*position \{([^}]*)\}\s*orientation \{([^}]*)\}')
_FIELD = re.compile(r"(\w+): (\S+)")


@dataclass(frozen=True, slots=True)
class RecordedPose:
    stamp_ns: int
    name: str
    x: float
    y: float
    yaw: float


def parse_recording(text: str) -> list[RecordedPose]:
    """The poses of a `gz topic -e` text recording, in recorded order; yaw from the quaternion (planar
    rotation about z: yaw = 2 atan2(qz, qw))."""
    out: list[RecordedPose] = []
    for block in _HEADER.split(text)[1:]:
        stamp = _STAMP.search(block)
        if stamp is None:
            raise ValueError("recording: a message without a header stamp")
        t_ns = int(stamp.group(1) or 0) * 1_000_000_000 + int(stamp.group(2) or 0)
        for pose in _POSE.finditer(block):
            pos = {k: float(v) for k, v in _FIELD.findall(pose.group(2))}
            ori = {k: float(v) for k, v in _FIELD.findall(pose.group(3))}
            yaw = 2.0 * math.atan2(ori.get("z", 0.0), ori.get("w", 1.0))
            out.append(RecordedPose(t_ns, pose.group(1), pos.get("x", 0.0), pos.get("y", 0.0), yaw))
    return out


@dataclass(frozen=True, slots=True)
class Deviation:
    """Agreement of one object: sample count and maximum deviations, with the stamp of the worst position."""

    name: str
    samples: int
    max_position_m: float
    max_position_stamp_ns: int | None
    max_yaw_rad: float

    @property
    def passed(self) -> bool:
        return (
            self.samples > 0
            and self.max_position_m <= POSITION_LIMIT_M
            and self.max_yaw_rad <= YAW_LIMIT_RAD
        )


def compare(
    poses: list[RecordedPose], objects: tuple[ParsedObject, ...]
) -> tuple[list[Deviation], list[str]]:
    """Per-object deviations from p(t) at the recorded stamps, and the recorded names absent from the SDF."""
    by_name = {o.id: o for o in objects}
    samples = dict.fromkeys(by_name, 0)
    position = dict.fromkeys(by_name, 0.0)
    position_at: dict[str, int | None] = dict.fromkeys(by_name, None)
    yaw = dict.fromkeys(by_name, 0.0)
    unknown: set[str] = set()
    for rec in poses:
        obj = by_name.get(rec.name)
        if obj is None:
            unknown.add(rec.name)
            continue
        p = pose_at(obj, rec.stamp_ns)
        dx, dy = rec.x - p.x, rec.y - p.y
        dp = math.sqrt(dx * dx + dy * dy)
        samples[rec.name] += 1
        if position_at[rec.name] is None or dp > position[rec.name]:
            position[rec.name], position_at[rec.name] = dp, rec.stamp_ns
        yaw[rec.name] = max(yaw[rec.name], abs(wrap(rec.yaw - p.yaw)))
    deviations = [
        Deviation(name, samples[name], position[name], position_at[name], yaw[name])
        for name in by_name
    ]
    return deviations, sorted(unknown)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m worldgen.agreement",
        description="compare a recorded pose topic with the reference evaluator (E0 §10, step 4)",
    )
    parser.add_argument(
        "--recording", required=True, help="text output of gz topic -e on the topic"
    )
    parser.add_argument("--world", required=True, help="the world SDF that was run")
    args = parser.parse_args(argv)
    plugin = objects_from_sdf(Path(args.world).read_text(encoding="utf-8"))
    poses = parse_recording(Path(args.recording).read_text(encoding="utf-8"))
    deviations, unknown = compare(poses, plugin.objects)
    stamps = sorted({p.stamp_ns for p in poses})
    if stamps:
        print(f"{len(stamps)} recorded stamps, {stamps[0]} ns to {stamps[-1]} ns")
    for d in deviations:
        print(
            f"{'PASS' if d.passed else 'FAIL'}  {d.name}: {d.samples} samples; max position "
            f"{d.max_position_m * 1e3:.6f} mm (at {d.max_position_stamp_ns} ns), max yaw "
            f"{d.max_yaw_rad * 1e3:.6f} mrad"
        )
    for name in unknown:
        print(f"FAIL  {name}: recorded but not in the SDF")
    print(f"pass limits: {POSITION_LIMIT_M * 1e3:g} mm, {YAW_LIMIT_RAD * 1e3:g} mrad")
    return 0 if all(d.passed for d in deviations) and not unknown else 1


if __name__ == "__main__":
    raise SystemExit(main())
