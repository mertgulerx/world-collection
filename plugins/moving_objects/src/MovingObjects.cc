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

#include "MovingObjects.hh"

#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <gz/common/Console.hh>
#include <gz/math/Pose3.hh>
#include <gz/math/Vector3.hh>
#include <gz/msgs/Utility.hh>
#include <gz/plugin/Register.hh>
#include <gz/sim/Util.hh>
#include <gz/sim/components/AngularVelocityCmd.hh>
#include <gz/sim/components/LinearVelocityCmd.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/sim/components/Physics.hh>
#include <gz/sim/components/Pose.hh>
#include <gz/sim/components/PoseCmd.hh>
#include <gz/sim/components/Static.hh>
#include <gz/transport/TopicUtils.hh>

namespace benchmark
{
namespace
{

constexpr char kLogPrefix[] = "[benchmark::MovingObjects] ";
constexpr std::int64_t kFormatVersion = 1;
constexpr char kDefaultTopic[] = "/benchmark/moving_objects/pose";
constexpr std::int64_t kDefaultPublishPeriodMs = 50;
constexpr double kMinSpeedMS = 0.3;
constexpr double kMaxSpeedMS = 1.0;
constexpr double kMaxDwellS = 30.0;
constexpr std::int64_t kNsPerMs = 1000000;
constexpr std::int64_t kNsPerS = 1000000000;

std::string trim(const std::string & text)
{
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) {
    return "";
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

/// Parses a whole string as a finite double (strtod is correctly rounded, like Python's float()).
std::optional<double> parse_double(const std::string & raw)
{
  const std::string text = trim(raw);
  if (text.empty()) {
    return std::nullopt;
  }
  errno = 0;
  char * end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  if (errno != 0 || end != text.c_str() + text.size() || !std::isfinite(value)) {
    return std::nullopt;
  }
  return value;
}

/// Parses a whole string as a base-10 signed 64-bit integer.
std::optional<std::int64_t> parse_int64(const std::string & raw)
{
  const std::string text = trim(raw);
  if (text.empty()) {
    return std::nullopt;
  }
  errno = 0;
  char * end = nullptr;
  const long long value = std::strtoll(text.c_str(), &end, 10);  // NOLINT(runtime/int)
  if (errno != 0 || end != text.c_str() + text.size()) {
    return std::nullopt;
  }
  return static_cast<std::int64_t>(value);
}

std::size_t count_children(const sdf::ElementPtr & parent, const std::string & name)
{
  std::size_t count = 0;
  for (sdf::ElementPtr child = parent->FindElement(name); child;
    child = child->GetNextElement(name))
  {
    ++count;
  }
  return count;
}

/// Reads the text of the single child <name>. Logs and returns nullopt when it is missing (and
/// required) or repeated; returns fallback when it is missing and a fallback is given.
std::optional<std::string> child_text(
  const sdf::ElementPtr & parent, const std::string & name, const std::string & context,
  const std::optional<std::string> & fallback = std::nullopt)
{
  const std::size_t count = count_children(parent, name);
  if (count == 0) {
    if (fallback) {
      return fallback;
    }
    gzerr << kLogPrefix << context << "field <" << name << "> is missing" << std::endl;
    return std::nullopt;
  }
  if (count > 1) {
    gzerr << kLogPrefix << context << "field <" << name << "> is given " << count << " times"
          << std::endl;
    return std::nullopt;
  }
  return trim(parent->FindElement(name)->Get<std::string>());
}

std::optional<double> child_double(
  const sdf::ElementPtr & parent, const std::string & name, const std::string & context)
{
  const auto text = child_text(parent, name, context);
  if (!text) {
    return std::nullopt;
  }
  const auto value = parse_double(*text);
  if (!value) {
    gzerr << kLogPrefix << context << "field <" << name << "> = '" << *text
          << "' is not a finite number" << std::endl;
  }
  return value;
}

std::optional<std::int64_t> child_int64(
  const sdf::ElementPtr & parent, const std::string & name, const std::string & context,
  const std::optional<std::string> & fallback = std::nullopt)
{
  const auto text = child_text(parent, name, context, fallback);
  if (!text) {
    return std::nullopt;
  }
  const auto value = parse_int64(*text);
  if (!value) {
    gzerr << kLogPrefix << context << "field <" << name << "> = '" << *text
          << "' is not a 64-bit integer" << std::endl;
  }
  return value;
}

std::optional<double> attribute_double(
  const sdf::ElementPtr & element, const std::string & name, const std::string & context)
{
  const sdf::ParamPtr attribute = element->GetAttribute(name);
  if (!attribute) {
    gzerr << kLogPrefix << context << "attribute '" << name << "' is missing" << std::endl;
    return std::nullopt;
  }
  const std::string text = attribute->GetAsString();
  const auto value = parse_double(text);
  if (!value) {
    gzerr << kLogPrefix << context << "attribute '" << name << "' = '" << text
          << "' is not a finite number" << std::endl;
  }
  return value;
}

void set_stamp(gz::msgs::Time * stamp, std::int64_t t_ns)
{
  stamp->set_sec(t_ns / kNsPerS);
  stamp->set_nsec(static_cast<std::int32_t>(t_ns % kNsPerS));
}

}  // namespace

void MovingObjects::Configure(
  const gz::sim::Entity & entity, const std::shared_ptr<const sdf::Element> & sdf,
  gz::sim::EntityComponentManager & /*ecm*/, gz::sim::EventManager & /*event_manager*/)
{
  world_ = entity;
  if (!parse(sdf->Clone())) {
    disable("invalid SDF parameters (see the messages above); no object will move");
    return;
  }
  publisher_ = node_.Advertise<gz::msgs::Pose_V>(topic_);
  if (!publisher_) {
    disable("cannot advertise topic '" + topic_ + "'; no object will move");
    return;
  }
  // Models are resolved at the first PreUpdate, not here: the order in which a world's SDF
  // plugins and its models are created is not guaranteed at Configure time.
  state_ = State::kUnresolved;
  gzmsg << kLogPrefix << "configured " << objects_.size() << " object(s), publishing on '"
        << topic_ << "' every " << publish_period_ms_ << " ms" << std::endl;
}

bool MovingObjects::parse(const sdf::ElementPtr & sdf)
{
  bool ok = true;
  const std::string plugin_context = "plugin: ";

  const auto version = child_int64(sdf, "format_version", plugin_context);
  if (version && *version != kFormatVersion) {
    gzerr << kLogPrefix << "plugin: field <format_version> = " << *version << " is not "
          << kFormatVersion << std::endl;
    ok = false;
  }
  ok = ok && version.has_value();

  const auto topic = child_text(sdf, "topic", plugin_context, std::string(kDefaultTopic));
  if (topic && !gz::transport::TopicUtils::IsValidTopic(*topic)) {
    gzerr << kLogPrefix << "plugin: field <topic> = '" << *topic
          << "' is not a valid gz-transport topic" << std::endl;
    ok = false;
  }
  ok = ok && topic.has_value();
  topic_ = topic.value_or("");

  const auto period = child_int64(
    sdf, "publish_period_ms", plugin_context, std::to_string(kDefaultPublishPeriodMs));
  if (period && *period <= 0) {
    gzerr << kLogPrefix << "plugin: field <publish_period_ms> = " << *period
          << " is not positive" << std::endl;
    ok = false;
  }
  ok = ok && period.has_value();
  publish_period_ms_ = period.value_or(0);
  publish_period_ns_ = publish_period_ms_ * kNsPerMs;

  const auto velocity_mode = child_text(sdf, "velocity_mode", plugin_context);
  if (velocity_mode) {
    if (*velocity_mode == "pose") {
      velocity_mode_ = VelocityMode::kPose;
    } else if (*velocity_mode == "pose_and_velocity") {
      velocity_mode_ = VelocityMode::kPoseAndVelocity;
    } else {
      gzerr << kLogPrefix << "plugin: field <velocity_mode> = '" << *velocity_mode
            << "' is not one of pose, pose_and_velocity" << std::endl;
      ok = false;
    }
  }
  ok = ok && velocity_mode.has_value();

  const auto turn_rate = child_double(sdf, "turn_rate", plugin_context);
  const bool turn_rate_ok = turn_rate && *turn_rate > 0.0;
  if (turn_rate && !turn_rate_ok) {
    gzerr << kLogPrefix << "plugin: field <turn_rate> = " << *turn_rate << " is not positive"
          << std::endl;
  }
  ok = ok && turn_rate_ok;

  // The seed is recorded only; the system draws nothing. It must still be an integer if given.
  if (count_children(sdf, "seed") > 0) {
    ok = child_int64(sdf, "seed", plugin_context).has_value() && ok;
  }

  if (count_children(sdf, "object") == 0) {
    gzerr << kLogPrefix << "plugin: no <object> element" << std::endl;
    return false;
  }

  std::set<std::string> names;
  std::size_t index = 0;
  for (sdf::ElementPtr object = sdf->FindElement("object"); object;
    object = object->GetNextElement("object"), ++index)
  {
    bool object_ok = true;
    const auto model = child_text(object, "model", "object #" + std::to_string(index) + ": ");
    if (!model) {
      ok = false;
      continue;
    }
    const std::string context = "object '" + *model + "': ";
    if (model->empty()) {
      gzerr << kLogPrefix << "object #" << index << ": field <model> is empty" << std::endl;
      ok = false;
      continue;
    }
    if (!names.insert(*model).second) {
      gzerr << kLogPrefix << context << "field <model> names a model already listed"
            << std::endl;
      object_ok = false;
    }

    // Objects are checked even after a plugin-level error so that every problem is reported.
    moving::PathSpec spec;
    spec.turn_rate_rad_s = turn_rate_ok ? *turn_rate : 1.0;

    const auto mode = child_text(object, "mode", context);
    if (mode && *mode == "loop") {
      spec.mode = moving::PathMode::kLoop;
    } else if (mode && *mode == "patrol") {
      spec.mode = moving::PathMode::kPatrol;
    } else {
      if (mode) {
        gzerr << kLogPrefix << context << "field <mode> = '" << *mode
              << "' is not one of loop, patrol" << std::endl;
      }
      object_ok = false;
    }

    const auto yaw_mode = child_text(object, "yaw_mode", context);
    if (yaw_mode && *yaw_mode == "fixed") {
      spec.yaw_mode = moving::YawMode::kFixed;
    } else if (yaw_mode && *yaw_mode == "heading") {
      spec.yaw_mode = moving::YawMode::kHeading;
    } else {
      if (yaw_mode) {
        gzerr << kLogPrefix << context << "field <yaw_mode> = '" << *yaw_mode
              << "' is not one of fixed, heading" << std::endl;
      }
      object_ok = false;
    }

    const auto speed = child_double(object, "speed", context);
    if (speed && !(*speed >= kMinSpeedMS && *speed <= kMaxSpeedMS)) {
      gzerr << kLogPrefix << context << "field <speed> = " << *speed << " m/s is outside ["
            << kMinSpeedMS << ", " << kMaxSpeedMS << "]" << std::endl;
      object_ok = false;
    }
    object_ok = object_ok && speed.has_value();
    spec.speed_m_s = speed.value_or(0.0);

    const auto yaw = child_double(object, "yaw", context);
    const auto z = child_double(object, "z", context);
    const auto phase_ns = child_int64(object, "phase_ns", context);
    const auto cycle_ns = child_int64(object, "cycle_ns", context);
    object_ok = object_ok && yaw && z && phase_ns && cycle_ns;
    spec.yaw_rad = yaw.value_or(0.0);

    std::size_t wp_index = 0;
    for (sdf::ElementPtr wp = object->FindElement("waypoint"); wp;
      wp = wp->GetNextElement("waypoint"), ++wp_index)
    {
      const std::string wp_context = context + "waypoint #" + std::to_string(wp_index) + ": ";
      const auto x = attribute_double(wp, "x", wp_context);
      const auto y = attribute_double(wp, "y", wp_context);
      const auto dwell = attribute_double(wp, "dwell", wp_context);
      if (dwell && !(*dwell >= 0.0 && *dwell <= kMaxDwellS)) {
        gzerr << kLogPrefix << wp_context << "attribute 'dwell' = " << *dwell
              << " s is outside [0, " << kMaxDwellS << "]" << std::endl;
        object_ok = false;
      }
      if (!(x && y && dwell)) {
        object_ok = false;
        continue;
      }
      spec.waypoints.push_back({*x, *y, *dwell});
    }
    if (wp_index < 2) {
      gzerr << kLogPrefix << context << "field <waypoint> needs at least two entries, got "
            << wp_index << std::endl;
      object_ok = false;
    }
    if (!object_ok || !turn_rate_ok) {
      ok = false;
      continue;
    }

    std::string error;
    auto timeline = moving::Timeline::build(spec, &error);
    if (!timeline) {
      gzerr << kLogPrefix << context << "field <waypoint>: " << error << std::endl;
      ok = false;
      continue;
    }
    if (timeline->cycle_ns() != *cycle_ns) {
      gzerr << kLogPrefix << context << "field <cycle_ns> = " << *cycle_ns
            << " differs from the recomputed cycle " << timeline->cycle_ns() << " ns"
            << std::endl;
      ok = false;
      continue;
    }
    if (*phase_ns < 0 || *phase_ns >= *cycle_ns) {
      gzerr << kLogPrefix << context << "field <phase_ns> = " << *phase_ns
            << " is outside [0, cycle_ns = " << *cycle_ns << ")" << std::endl;
      ok = false;
      continue;
    }
    objects_.push_back({*model, gz::sim::kNullEntity, *z, *phase_ns, std::move(*timeline)});
  }
  return ok;
}

bool MovingObjects::resolve(gz::sim::EntityComponentManager & ecm)
{
  bool ok = true;

  const auto * physics = ecm.Component<gz::sim::components::Physics>(world_);
  if (physics == nullptr) {
    gzerr << kLogPrefix << "plugin: the world has no physics settings to check "
          << "<publish_period_ms> against" << std::endl;
    ok = false;
  } else {
    const std::int64_t step_ns = moving::round_ns(physics->Data().MaxStepSize() * 1e9);
    if (step_ns <= 0 || publish_period_ns_ % step_ns != 0) {
      gzerr << kLogPrefix << "plugin: field <publish_period_ms> = " << publish_period_ms_
            << " is not a multiple of the physics step of " << step_ns << " ns" << std::endl;
      ok = false;
    }
  }

  for (Object & object : objects_) {
    std::size_t top_level = 0;
    std::size_t nested = 0;
    gz::sim::Entity found = gz::sim::kNullEntity;
    ecm.Each<gz::sim::components::Model, gz::sim::components::Name>(
      [&](const gz::sim::Entity & candidate, const gz::sim::components::Model *,
      const gz::sim::components::Name * name) -> bool
      {
        if (name->Data() != object.model_name) {
          return true;
        }
        const auto parent = ecm.ComponentData<gz::sim::components::ParentEntity>(candidate);
        if (parent && *parent == world_) {
          ++top_level;
          found = candidate;
        } else {
          ++nested;
        }
        return true;
      });
    const std::string context = "object '" + object.model_name + "': field <model>: ";
    if (top_level + nested == 0) {
      gzerr << kLogPrefix << context << "no model with this name exists" << std::endl;
      ok = false;
      continue;
    }
    if (top_level + nested > 1) {
      gzerr << kLogPrefix << context << "the name is duplicated (" << top_level
            << " top-level, " << nested << " nested model(s))" << std::endl;
      ok = false;
      continue;
    }
    if (nested > 0) {
      gzerr << kLogPrefix << context << "the model is nested; only top-level models can be moved"
            << std::endl;
      ok = false;
      continue;
    }
    const auto is_static = ecm.ComponentData<gz::sim::components::Static>(found);
    if (is_static && *is_static) {
      gzerr << kLogPrefix << context << "the model is static" << std::endl;
      ok = false;
      continue;
    }
    object.entity = found;
  }

  if (!ok) {
    disable("cannot resolve the moving objects (see the messages above); no object will move");
    return false;
  }

  message_.Clear();
  for (const Object & object : objects_) {
    message_.add_pose()->set_name(object.model_name);
  }
  state_ = State::kActive;
  return true;
}

void MovingObjects::disable(const std::string & reason)
{
  gzerr << kLogPrefix << "system disabled: " << reason << std::endl;
  state_ = State::kDisabled;
}

void MovingObjects::PreUpdate(
  const gz::sim::UpdateInfo & info, gz::sim::EntityComponentManager & ecm)
{
  if (state_ == State::kUnresolved && !resolve(ecm)) {
    return;
  }
  if (state_ != State::kActive || info.paused) {
    return;
  }
  const std::int64_t t_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime).count();
  const bool command_velocity = velocity_mode_ == VelocityMode::kPoseAndVelocity;
  const std::int64_t dt_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(info.dt).count();
  for (const Object & object : objects_) {
    const moving::PathSample sample = object.timeline.evaluate(t_ns, object.phase_ns);
    if (!command_velocity || dt_ns <= 0) {
      // The Physics system applies a model's WorldPoseCmd to the root link of its free group; in
      // `pose` mode the body is then at p(t) when PostUpdate reads it (agreement test, plan §10).
      ecm.SetComponentData<gz::sim::components::WorldPoseCmd>(
        object.entity, gz::math::Pose3d(sample.x_m, sample.y_m, object.z_m, 0.0, 0.0,
        sample.yaw_rad));
      if (command_velocity) {
        // A zero-length unpaused step (not expected with a fixed physics step) has no average
        // velocity: clear the velocity commands so a previous step's values are not integrated.
        ecm.SetComponentData<gz::sim::components::LinearVelocityCmd>(
          object.entity, gz::math::Vector3d::Zero);
        ecm.SetComponentData<gz::sim::components::AngularVelocityCmd>(
          object.entity, gz::math::Vector3d::Zero);
      }
      continue;
    }
    // `pose_and_velocity`: Physics applies the pose command and then integrates the velocity over
    // the step, so commanding p(t) with the path velocity lands one step ahead (measured: 5–8 mm at
    // 0.5–0.8 m/s). Command the pose at the start of the step, p(t − dt), and the average velocity
    // over the step, (p(t) − p(t − dt)) / dt; the body then ends the step at p(t), also when a phase
    // boundary falls inside the step. Velocity commands are in the model frame (roll and pitch 0).
    const moving::PathSample start = object.timeline.evaluate(t_ns - dt_ns, object.phase_ns);
    const double dt_s = static_cast<double>(dt_ns) * 1e-9;
    const double vx = (sample.x_m - start.x_m) / dt_s;
    const double vy = (sample.y_m - start.y_m) / dt_s;
    // Yaws are not re-wrapped inside a cycle but restart at the cycle wrap: take the short way.
    const double dyaw = moving::wrap_angle(sample.yaw_rad - start.yaw_rad);
    ecm.SetComponentData<gz::sim::components::WorldPoseCmd>(
      object.entity, gz::math::Pose3d(start.x_m, start.y_m, object.z_m, 0.0, 0.0, start.yaw_rad));
    const double c = std::cos(start.yaw_rad);
    const double s = std::sin(start.yaw_rad);
    ecm.SetComponentData<gz::sim::components::LinearVelocityCmd>(
      object.entity, gz::math::Vector3d(c * vx + s * vy, -s * vx + c * vy, 0.0));
    ecm.SetComponentData<gz::sim::components::AngularVelocityCmd>(
      object.entity, gz::math::Vector3d(0.0, 0.0, dyaw / dt_s));
  }
}

void MovingObjects::PostUpdate(
  const gz::sim::UpdateInfo & info, const gz::sim::EntityComponentManager & ecm)
{
  if (state_ != State::kActive || info.paused) {
    return;
  }
  const std::int64_t t_ns =
    std::chrono::duration_cast<std::chrono::nanoseconds>(info.simTime).count();
  if (t_ns % publish_period_ns_ != 0) {
    return;
  }
  set_stamp(message_.mutable_header()->mutable_stamp(), t_ns);
  for (std::size_t i = 0; i < objects_.size(); ++i) {
    const gz::math::Pose3d pose = gz::sim::worldPose(objects_[i].entity, ecm);
    gz::msgs::Pose * out = message_.mutable_pose(static_cast<int>(i));
    set_stamp(out->mutable_header()->mutable_stamp(), t_ns);
    gz::msgs::Set(out->mutable_position(), pose.Pos());
    gz::msgs::Set(out->mutable_orientation(), pose.Rot());
  }
  publisher_.Publish(message_);
}

}  // namespace benchmark

GZ_ADD_PLUGIN(
  benchmark::MovingObjects, gz::sim::System, benchmark::MovingObjects::ISystemConfigure,
  benchmark::MovingObjects::ISystemPreUpdate, benchmark::MovingObjects::ISystemPostUpdate)
