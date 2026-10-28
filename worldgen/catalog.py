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

"""The buildable worlds, by id. Each entry returns a fresh World definition."""

from __future__ import annotations

from collections.abc import Callable

from .diagnostic import (
    b1_square_loop,
    b2_narrow_passage,
    b3_dead_end_comb,
    b4_office_strip,
    b5_nested_pockets,
    b6_open_hall,
    b7_clutter_field,
    b8_zigzag,
)
from .families import aisles, forests, mazes, offices, urban
from .large import ls1_gargantua
from .moving_worlds import e1_office_strip_moving, e3_warehouse_moving
from .named import r3_aces3_austin, r8_industrial_warehouse, r10_santa_maria_del_mar
from .pilot import ps1_racetrack_office, ps2_lobby_wing
from .world import FamilyInfo, World

WORLDS: dict[str, Callable[[], World]] = {
    "SC1": b1_square_loop.build,
    "SC2a": b2_narrow_passage.build_a,
    "SC2b": b2_narrow_passage.build_b,
    "SC3": b3_dead_end_comb.build,
    "SC4": b4_office_strip.build,
    # B5 alcove hall was dropped (decision 48); B5 is now the nested-pockets world (decision 50)
    "SC5": b5_nested_pockets.build,
    "SC6": b6_open_hall.build,
    "SC7a": b7_clutter_field.build_a,
    "SC7b": b7_clutter_field.build_b,
    "SC8": b8_zigzag.build,
    "P1": ps1_racetrack_office.build,
    "P2": ps2_lobby_wing.build,
    **mazes.builders(),
    **offices.builders(),
    **aisles.builders(),
    **forests.builders(),
    **urban.builders(),
    "RP1": r3_aces3_austin.build,
    "RP2": r8_industrial_warehouse.build,
    "RP3": r10_santa_maria_del_mar.build,
    # E3-low is not built (maintainer, 18 October 2026; report §9 decision 46)
    "MO2": e3_warehouse_moving.build_high,
    # E1-low is not built (maintainer, 18 October 2026; report §9 decision 45)
    "MO1": e1_office_strip_moving.build_high,
    "LS1": ls1_gargantua.build,
}


FAMILIES: dict[str, FamilyInfo] = {
    "F1": mazes.FAMILY_INFO,
    "F2": offices.FAMILY_INFO,
    "F3": aisles.FAMILY_INFO,
    "F4": forests.FAMILY_INFO,
    "F5": urban.FAMILY_INFO,
}


def get(world_id: str) -> World:
    try:
        factory = WORLDS[world_id]
    except KeyError:
        raise SystemExit(
            f"unknown world {world_id!r}; known: {', '.join(sorted(WORLDS))}"
        ) from None
    world = factory()
    world.validate()
    return world
