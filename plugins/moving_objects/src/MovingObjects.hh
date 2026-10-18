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

// World system that moves scripted solid objects as a pure function of simulation time and
// publishes their actual world poses (plan docs/plans/worlds/E0-moving-object-system.md).

#ifndef BENCHMARK_MOVING_OBJECTS__MOVING_OBJECTS_HH_
#define BENCHMARK_MOVING_OBJECTS__MOVING_OBJECTS_HH_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gz/msgs/pose_v.pb.h>

#include <gz/sim/Entity.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/EventManager.hh>
#include <gz/sim/System.hh>
#include <gz/transport/Node.hh>
#include <sdf/Element.hh>

#include "Kinematics.hh"

namespace benchmark
{

class MovingObjects
  : public gz::sim::System,
  public gz::sim::ISystemConfigure,
  public gz::sim::ISystemPreUpdate,
  public gz::sim::ISystemPostUpdate
{
public:
  void Configure(
    const gz::sim::Entity & entity, const std::shared_ptr<const sdf::Element> & sdf,
    gz::sim::EntityComponentManager & ecm, gz::sim::EventManager & event_manager) override;

  void PreUpdate(const gz::sim::UpdateInfo & info, gz::sim::EntityComponentManager & ecm) override;

  void PostUpdate(
    const gz::sim::UpdateInfo & info, const gz::sim::EntityComponentManager & ecm) override;

private:
  enum class State { kDisabled, kUnresolved, kActive };
  enum class VelocityMode { kPose, kPoseAndVelocity };

  struct Object
  {
    std::string model_name;
    gz::sim::Entity entity{gz::sim::kNullEntity};
    double z_m{0.0};
    std::int64_t phase_ns{0};
    moving::Timeline timeline;
  };

  bool parse(const sdf::ElementPtr & sdf);
  bool resolve(gz::sim::EntityComponentManager & ecm);
  void disable(const std::string & reason);

  State state_{State::kDisabled};
  gz::sim::Entity world_{gz::sim::kNullEntity};
  VelocityMode velocity_mode_{VelocityMode::kPose};
  std::string topic_;
  std::int64_t publish_period_ms_{0};
  std::int64_t publish_period_ns_{0};
  std::vector<Object> objects_;

  gz::transport::Node node_;
  gz::transport::Node::Publisher publisher_;
  gz::msgs::Pose_V message_;  // sized once at resolve, reused on every publish
};

}  // namespace benchmark

#endif  // BENCHMARK_MOVING_OBJECTS__MOVING_OBJECTS_HH_
