# Moving-object system

`benchmark::MovingObjects` is a Gazebo Harmonic (gz-sim 8) world system. It moves solid objects along
scripted loops or patrols as a pure function of simulation time, and publishes their actual world poses as
`gz.msgs.Pose_V` on `/benchmark/moving_objects/pose`, stamped with simulation time. The design is plan
`docs/plans/worlds/E0-moving-object-system.md`; the `<plugin>` block of each world is written by `worldgen`.

## Build
Needs ROS 2 Jazzy with its Gazebo vendor packages (`gz-sim8`, `gz-plugin2`, `gz-transport13`, `gz-msgs10`).

```bash
source /opt/ros/jazzy/setup.bash
cmake -S world_collection/plugins/moving_objects -B world_collection/plugins/moving_objects/build
cmake --build world_collection/plugins/moving_objects/build
cmake --install world_collection/plugins/moving_objects/build
```

The install step writes `libbenchmark-moving-objects-system.so` to `world_collection/plugins/lib/`.

## Use
Put the library directory on the system plugin path before starting Gazebo:

```bash
export GZ_SIM_SYSTEM_PLUGIN_PATH="$PWD/world_collection/plugins/lib:${GZ_SIM_SYSTEM_PLUGIN_PATH}"
```

The world loads it with `<plugin filename="benchmark-moving-objects-system" name="benchmark::MovingObjects">`.
Every moved object must be a top-level, non-static model. Any invalid parameter disables the whole system with
an error that names the object and the field, and then no object moves.

## Files
- `src/Kinematics.hh`: the deterministic path kinematics (header-only, no Gazebo types). The Python evaluator
  in `worldgen/moving.py` implements the same rules, listed at the top of this file.
- `src/MovingObjects.hh`, `src/MovingObjects.cc`: SDF parsing and validation, pose commands, publishing.
