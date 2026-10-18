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

# E0 moving-object system (docs/plans/worlds/E0-moving-object-system.md). Plain CMake; the
# Gazebo Harmonic libraries come from the ROS 2 Jazzy vendor packages, whose prefixes
# /opt/ros/jazzy/setup.bash adds to CMAKE_PREFIX_PATH.
cmake_minimum_required(VERSION 3.16)
project(benchmark_moving_objects LANGUAGES CXX)

find_package(gz-sim8 REQUIRED)
find_package(gz-plugin2 REQUIRED COMPONENTS register)
find_package(gz-transport13 REQUIRED)
find_package(gz-msgs10 REQUIRED)

# gz-sim resolves <plugin filename="benchmark-moving-objects-system"> to
# libbenchmark-moving-objects-system.so on GZ_SIM_SYSTEM_PLUGIN_PATH.
add_library(benchmark-moving-objects-system SHARED src/MovingObjects.cc)
set_target_properties(benchmark-moving-objects-system PROPERTIES
  CXX_STANDARD 17
  CXX_STANDARD_REQUIRED ON
  CXX_EXTENSIONS OFF)
# Fixed flags, independent of CMAKE_BUILD_TYPE, no -ffast-math and no fused multiply-add
# contraction (Python never fuses): the kinematics must match the Python evaluator bit for bit.
target_compile_options(benchmark-moving-objects-system PRIVATE
  -O2 -ffp-contract=off -Wall -Wextra -Wpedantic -Werror)
target_link_libraries(benchmark-moving-objects-system PRIVATE
  gz-sim8::gz-sim8
  gz-plugin2::register
  gz-transport13::gz-transport13
  gz-msgs10::gz-msgs10)

set(MOVING_OBJECTS_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../lib"
  CACHE PATH "Directory that receives the plugin library (GZ_SIM_SYSTEM_PLUGIN_PATH)")
install(TARGETS benchmark-moving-objects-system
  LIBRARY DESTINATION "${MOVING_OBJECTS_LIB_DIR}")
