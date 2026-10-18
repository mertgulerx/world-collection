/*
Copyright 2026 Mert Güler

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Deterministic path kinematics of the E0 moving-object system
// (docs/plans/worlds/E0-moving-object-system.md, section 3.4). Header-only and free of Gazebo
// types so that the same code can be compiled outside the plugin. The Python evaluator
// (world_collection/worldgen/moving.py) implements the same definition; every rule below that the
// plan leaves open is stated here and must be matched there. Bit-for-bit agreement needs the
// build without FMA contraction (CMakeLists.txt) and the same libm atan2 for heading yaws; atan2 is
// not guaranteed correctly rounded across C libraries, and the plugin's cycle_ns equality check
// against the SDF value is the guard that catches a resulting duration change.
//
// Timeline rules (the cycle starts, tau = 0, at w_0 at the start of the move w_0 -> w_1):
// 1. Segments. loop: w_0->w_1, ..., w_{n-2}->w_{n-1}, w_{n-1}->w_0. patrol: forward pass
//    w_0->...->w_{n-1}, then backward pass w_{n-1}->...->w_0.
// 2. Segment yaw. fixed: the object's yaw. heading: atan2(dy, dx) of the segment's own direction;
//    on the patrol backward pass, the forward direction of the same pair of waypoints, i.e.
//    atan2(y_k - y_{k-1}, x_k - x_{k-1}) for the backward segment w_k -> w_{k-1}.
// 3. Phases per segment, in this order: move along the segment; on arrival at its end waypoint b,
//    a turn from this segment's yaw to the next segment's yaw (next of the last segment is the
//    first segment); then the dwell d_b. Turns exist only in heading mode, with
//    delta = wrap(yaw_next - yaw_this) in (-pi, pi]. In a loop this includes the turn at w_0
//    after w_{n-1}->w_0, at the end of the cycle, so the yaw at tau = 0 is the heading of
//    w_0->w_1. In a patrol, the turn delta at both endpoints is exactly 0 by rule 2 (reversal
//    without turning); intermediate waypoints turn on both passes and dwell on each arrival, the
//    endpoints w_{n-1} and w_0 dwell once per cycle.
// 4. Durations in integer nanoseconds, each rounded once with round-half-to-even on the exact
//    double (std::nearbyint here, Python's round() there): move round(l / v * 1e9) with
//    l = sqrt(dx * dx + dy * dy); turn round(|delta| / omega * 1e9); dwell round(d * 1e9).
//    A phase whose rounded duration is 0 is omitted (a zero dwell, a zero turn).
// 5. Pose at tau: the phase is the last one whose start <= tau (binary search over cumulative
//    starts); s = tau - start, f = double(s) / double(duration). Move: x = x0 + f * (x1 - x0),
//    y likewise, yaw = segment yaw. Turn: position of the waypoint, yaw = yaw_this + f * delta
//    (not re-wrapped). Dwell: position of the waypoint; yaw = the next segment's yaw when the
//    wrapped delta at this waypoint is non-zero (heading mode), else the arriving segment's yaw.
// 6. tau = (t + phase) mod T in 64-bit integers.

#ifndef BENCHMARK_MOVING_OBJECTS__KINEMATICS_HH_
#define BENCHMARK_MOVING_OBJECTS__KINEMATICS_HH_

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace benchmark
{
namespace moving
{

/// Shortest segment the plan allows, in metres (exclusive bound).
constexpr double kMinSegmentLengthM = 0.05;
/// pi as the IEEE double nearest to it (equal to Python's math.pi).
constexpr double kPi = 3.141592653589793;

enum class PathMode { kLoop, kPatrol };
enum class YawMode { kFixed, kHeading };
enum class PhaseKind { kMove, kTurn, kDwell };

struct Waypoint
{
  double x_m;
  double y_m;
  double dwell_s;
};

/// Validated path definition of one object (bounds are checked by the caller).
struct PathSpec
{
  std::vector<Waypoint> waypoints;
  PathMode mode{PathMode::kLoop};
  YawMode yaw_mode{YawMode::kFixed};
  double speed_m_s{0.0};
  double yaw_rad{0.0};
  double turn_rate_rad_s{1.0};
};

struct Phase
{
  PhaseKind kind;
  std::int64_t start_ns;
  std::int64_t duration_ns;
  double x0_m;       // position at the start of the phase
  double y0_m;
  double dx_m;       // displacement over a move phase, 0 otherwise
  double dy_m;
  double yaw0_rad;   // yaw at the start of the phase
  double dyaw_rad;   // yaw change over a turn phase, 0 otherwise
  double vx_m_s;     // world-frame path velocity of a move phase, 0 otherwise (not commanded)
  double vy_m_s;
  double yaw_rate_rad_s;  // signed turn rate of a turn phase, 0 otherwise
};

/// Pose and path velocity (world frame) at one instant. The velocities match Python's
/// velocity_at; the plugin does not command them (pose_and_velocity mode commands the average
/// velocity over the step, see MovingObjects::PreUpdate).
struct PathSample
{
  double x_m;
  double y_m;
  double yaw_rad;
  double vx_m_s;
  double vy_m_s;
  double yaw_rate_rad_s;
};

/// Wraps an angle to (-pi, pi]. std::remainder is exact (IEEE remainder), as is Python's
/// math.remainder, so both languages give the same double.
inline double wrap_angle(double angle_rad)
{
  double r = std::remainder(angle_rad, 2.0 * kPi);
  if (r <= -kPi) {
    r += 2.0 * kPi;
  }
  return r;
}

/// Rounds a non-negative nanosecond value half to even, like Python's round().
inline std::int64_t round_ns(double value_ns)
{
  return static_cast<std::int64_t>(std::nearbyint(value_ns));
}

class Timeline
{
public:
  /// Builds the timeline, or returns std::nullopt with a reason in *error.
  static std::optional<Timeline> build(const PathSpec & spec, std::string * error)
  {
    const std::size_t n = spec.waypoints.size();
    if (n < 2) {
      *error = "needs at least two waypoints";
      return std::nullopt;
    }
    if (!(spec.speed_m_s > 0.0) || !(spec.turn_rate_rad_s > 0.0)) {
      *error = "speed and turn rate must be positive";
      return std::nullopt;
    }

    struct Segment
    {
      std::size_t a;
      std::size_t b;
      double yaw_rad;
    };
    std::vector<Segment> segments;
    const auto heading = [&spec](std::size_t from, std::size_t to) {
        return std::atan2(
          spec.waypoints[to].y_m - spec.waypoints[from].y_m,
          spec.waypoints[to].x_m - spec.waypoints[from].x_m);
      };
    const bool use_heading = spec.yaw_mode == YawMode::kHeading;
    if (spec.mode == PathMode::kLoop) {
      for (std::size_t i = 0; i < n; ++i) {
        const std::size_t j = (i + 1) % n;
        segments.push_back({i, j, use_heading ? heading(i, j) : spec.yaw_rad});
      }
    } else {
      for (std::size_t i = 0; i + 1 < n; ++i) {
        segments.push_back({i, i + 1, use_heading ? heading(i, i + 1) : spec.yaw_rad});
      }
      for (std::size_t i = n - 1; i >= 1; --i) {
        // Backward pass keeps the forward yaw of the same waypoint pair.
        segments.push_back({i, i - 1, use_heading ? heading(i - 1, i) : spec.yaw_rad});
      }
    }

    Timeline timeline;
    std::int64_t t_ns = 0;
    const auto append = [&timeline, &t_ns](Phase phase) {
        if (phase.duration_ns <= 0) {
          return;
        }
        phase.start_ns = t_ns;
        t_ns += phase.duration_ns;
        timeline.starts_ns_.push_back(phase.start_ns);
        timeline.phases_.push_back(phase);
      };

    for (std::size_t k = 0; k < segments.size(); ++k) {
      const Segment & seg = segments[k];
      const Segment & next = segments[(k + 1) % segments.size()];
      const Waypoint & wa = spec.waypoints[seg.a];
      const Waypoint & wb = spec.waypoints[seg.b];
      const double dx = wb.x_m - wa.x_m;
      const double dy = wb.y_m - wa.y_m;
      const double length_m = std::sqrt(dx * dx + dy * dy);
      if (!(length_m > kMinSegmentLengthM)) {
        *error = "segment w_" + std::to_string(seg.a) + " -> w_" + std::to_string(seg.b) +
          " is not longer than 0.05 m";
        return std::nullopt;
      }
      append(
        {PhaseKind::kMove, 0, round_ns(length_m / spec.speed_m_s * 1e9), wa.x_m, wa.y_m, dx, dy,
          seg.yaw_rad, 0.0, dx / length_m * spec.speed_m_s, dy / length_m * spec.speed_m_s, 0.0});

      double yaw_after = seg.yaw_rad;
      if (use_heading) {
        const double delta = wrap_angle(next.yaw_rad - seg.yaw_rad);
        if (delta != 0.0) {
          append(
            {PhaseKind::kTurn, 0, round_ns(std::fabs(delta) / spec.turn_rate_rad_s * 1e9),
              wb.x_m, wb.y_m, 0.0, 0.0, seg.yaw_rad, delta, 0.0, 0.0,
              delta > 0.0 ? spec.turn_rate_rad_s : -spec.turn_rate_rad_s});
          yaw_after = next.yaw_rad;
        }
      }
      append(
        {PhaseKind::kDwell, 0, round_ns(wb.dwell_s * 1e9), wb.x_m, wb.y_m, 0.0, 0.0, yaw_after,
          0.0, 0.0, 0.0, 0.0});
    }
    timeline.cycle_ns_ = t_ns;
    return timeline;
  }

  /// Sum of all phase durations: the cycle time T in nanoseconds.
  std::int64_t cycle_ns() const {return cycle_ns_;}

  const std::vector<Phase> & phases() const {return phases_;}

  /// Commanded pose at simulation time t_ns for a phase offset 0 <= phase_ns < T.
  PathSample evaluate(std::int64_t t_ns, std::int64_t phase_ns) const
  {
    std::int64_t tau = t_ns % cycle_ns_;
    if (tau < 0) {
      tau += cycle_ns_;
    }
    tau = (tau + phase_ns) % cycle_ns_;
    const auto it = std::upper_bound(starts_ns_.begin(), starts_ns_.end(), tau);
    const Phase & ph = phases_[static_cast<std::size_t>(it - starts_ns_.begin()) - 1];
    const double f = static_cast<double>(tau - ph.start_ns) /
      static_cast<double>(ph.duration_ns);
    switch (ph.kind) {
      case PhaseKind::kMove:
        return {ph.x0_m + f * ph.dx_m, ph.y0_m + f * ph.dy_m, ph.yaw0_rad, ph.vx_m_s, ph.vy_m_s,
          0.0};
      case PhaseKind::kTurn:
        return {ph.x0_m, ph.y0_m, ph.yaw0_rad + f * ph.dyaw_rad, 0.0, 0.0, ph.yaw_rate_rad_s};
      case PhaseKind::kDwell:
      default:
        return {ph.x0_m, ph.y0_m, ph.yaw0_rad, 0.0, 0.0, 0.0};
    }
  }

private:
  Timeline() = default;

  std::vector<Phase> phases_;
  std::vector<std::int64_t> starts_ns_;
  std::int64_t cycle_ns_{0};
};

}  // namespace moving
}  // namespace benchmark

#endif  // BENCHMARK_MOVING_OBJECTS__KINEMATICS_HH_
