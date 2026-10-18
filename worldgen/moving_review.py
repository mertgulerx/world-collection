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

"""Review images of the moving objects (E0, T0 §8): `review/paths.png` and `review/timelapse.png`.

Both are computed from the path definition with the reference evaluator of `moving.py`; no simulator runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.axes import Axes  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

from . import WORLDGEN_VERSION  # noqa: E402
from .checks import Analysis  # noqa: E402
from .moving import (  # noqa: E402
    Mode,
    MovingDefinition,
    MovingObject,
    YawMode,
    footprint_polygon,
    pose_at,
)
from .plan_plot import FREE, OUTSIDE, SOLID, SPAWN  # noqa: E402
from .world import World  # noqa: E402

TIMES_S = (0.0, 10.0, 30.0, 60.0, 300.0, 900.0)
TRAIL_S = 5.0
DOORWAY = "#7a4fbf"


Colour = tuple[float, float, float, float]


def _colour(k: int) -> Colour:
    r, g, b, alpha = plt.get_cmap("tab10")(k % 10)
    return r, g, b, alpha


def _background(ax: Axes, a: Analysis) -> None:
    g = a.grid
    rgb = np.empty((g.ny, g.nx, 3))
    rgb[:] = OUTSIDE
    rgb[a.occupied] = SOLID
    rgb[a.reachable] = FREE
    extent = (g.x0, g.x0 + g.nx * g.h, g.y0, g.y0 + g.ny * g.h)
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest", zorder=0)
    ax.set_xlim(extent[0], extent[1])
    ax.set_ylim(extent[2], extent[3])
    ax.set_aspect("equal")


def _doorways(ax: Axes, defn: MovingDefinition) -> None:
    for d in defn.doorways:
        ax.add_patch(Polygon(d.polygon, closed=True, fc=DOORWAY, ec=DOORWAY, alpha=0.25, zorder=2))
        ax.annotate(
            d.id, (d.cx, d.cy), ha="center", va="center", fontsize=6, color=DOORWAY, zorder=2
        )


def _footprint(ax: Axes, obj: MovingObject, t_ns: int, colour: Colour, label: bool) -> None:
    p = pose_at(obj, t_ns)
    ax.add_patch(
        Polygon(footprint_polygon(obj, p), closed=True, fc=colour, ec="black", lw=0.6, zorder=6)
    )
    if obj.object_type.yaw_mode is YawMode.HEADING:
        ax.plot(
            [p.x, p.x + 0.6 * math.cos(p.yaw)],
            [p.y, p.y + 0.6 * math.sin(p.yaw)],
            color="black",
            lw=0.8,
            zorder=7,
        )
    if label:
        ax.annotate(
            obj.id, (p.x, p.y), xytext=(5, 5), textcoords="offset points", fontsize=7, zorder=7
        )


def paths_png(world: World, a: Analysis, path: Path, width_px: int = 1600) -> None:
    """Every object's path (direction arrows, waypoints with dwell labels), footprints at t = 0, doorway regions
    and the spawn exclusion discs (2.0 m at t = 0, 1.0 m always)."""
    defn = world.moving
    assert defn is not None
    g = a.grid
    span_x, span_y = g.nx * g.h, g.ny * g.h
    dpi = 100
    fig_w = width_px / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * span_y / span_x + 0.9), dpi=dpi)
    _background(ax, a)
    _doorways(ax, defn)
    lim = defn.limits
    for s in world.spawns:
        ax.add_patch(
            Circle(
                (s.x, s.y),
                lim.spawn_exclusion_t0_m,
                fill=False,
                ec=SPAWN,
                ls="--",
                lw=1.0,
                zorder=4,
            )
        )
        ax.add_patch(
            Circle((s.x, s.y), lim.spawn_clearance_always_m, fill=False, ec=SPAWN, lw=1.2, zorder=4)
        )
        ax.plot([s.x], [s.y], "o", color=SPAWN, ms=4, zorder=4)
        ax.annotate(
            s.id, (s.x, s.y), xytext=(5, -10), textcoords="offset points", fontsize=8, color=SPAWN
        )
    for k, obj in enumerate(defn.objects):
        colour = _colour(k)
        wps = list(obj.waypoints)
        closed = wps + [wps[0]] if obj.mode is Mode.LOOP else wps
        xs, ys = [w.x for w in closed], [w.y for w in closed]
        ax.plot(xs, ys, color=colour, lw=1.6, zorder=5)
        for w0, w1 in zip(closed[:-1], closed[1:], strict=True):
            mx, my = (w0.x + w1.x) / 2, (w0.y + w1.y) / 2
            dx, dy = w1.x - w0.x, w1.y - w0.y
            length = math.sqrt(dx * dx + dy * dy)
            ax.annotate(
                "",
                (mx + 0.25 * dx / length, my + 0.25 * dy / length),
                (mx - 0.25 * dx / length, my - 0.25 * dy / length),
                arrowprops={
                    "arrowstyle": "<|-|>" if obj.mode is Mode.PATROL else "-|>",
                    "color": colour,
                    "lw": 1.4,
                },
                zorder=5,
            )
        for w in wps:
            ax.plot([w.x], [w.y], "o", color=colour, ms=4, zorder=5)
            if w.dwell_s > 0:
                ax.annotate(
                    f"{w.dwell_s:g} s",
                    (w.x, w.y),
                    xytext=(4, -9),
                    textcoords="offset points",
                    fontsize=7,
                    color=colour,
                )
        _footprint(ax, obj, 0, colour, True)
    ax.set_xlabel("x (m, east)")
    ax.set_ylabel("y (m, north)")
    ax.set_title(
        f"{world.id} {world.title}: {defn.count} moving objects ({defn.density}), paths and footprints at t = 0 s"
        f"   worldgen {WORLDGEN_VERSION}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def timelapse_png(
    world: World, a: Analysis, path: Path, times_s: tuple[float, ...] = TIMES_S
) -> None:
    """A 2 × 3 sheet of the footprints at the given simulation times, each with a 5 s trail of its centre."""
    defn = world.moving
    assert defn is not None
    g = a.grid
    aspect = (g.ny * g.h) / (g.nx * g.h)
    cols = 3
    rows = math.ceil(len(times_s) / cols)
    fig, axes = plt.subplots(
        rows, cols, figsize=(6.0 * cols, 6.0 * aspect * rows + 0.8), squeeze=False, dpi=100
    )
    for ax in axes.flat:
        ax.set_axis_off()
    for panel, t_s in enumerate(times_s):
        ax = axes.flat[panel]
        ax.set_axis_on()
        _background(ax, a)
        _doorways(ax, defn)
        t_ns = round(t_s * 1e9)
        for k, obj in enumerate(defn.objects):
            colour = _colour(k)
            trail = [
                pose_at(obj, max(0, t_ns - round(d * 1e8)))
                for d in range(round(TRAIL_S * 10), -1, -1)
            ]
            ax.plot(
                [p.x for p in trail],
                [p.y for p in trail],
                color=colour,
                lw=1.2,
                alpha=0.8,
                zorder=5,
            )
            _footprint(ax, obj, t_ns, colour, panel == 0)
        for s in world.spawns:
            ax.plot([s.x], [s.y], "o", color=SPAWN, ms=4, zorder=4)
        ax.set_title(f"t = {t_s:g} s", fontsize=11)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{world.id} {world.title}: object positions with {TRAIL_S:g} s trails   worldgen {WORLDGEN_VERSION}",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)
