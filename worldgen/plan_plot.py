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

"""Top-down plan image `review/plan.png` (T0 §8)."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, FancyArrow, Polygon  # noqa: E402

from . import WORLDGEN_VERSION  # noqa: E402
from .checks import Analysis  # noqa: E402
from .descriptors import VERSION as DESCRIPTORS_VERSION  # noqa: E402
from .descriptors import regions  # noqa: E402
from .raster import EIGHT, skeleton_nodes  # noqa: E402
from .world import Kind, PlanKey, Primitive, Role, World  # noqa: E402

FREE = (0.95, 0.95, 0.92)
SOLID = (0.29, 0.29, 0.29)
SEALED = (0.85, 0.25, 0.25)
OUTSIDE = (1.0, 1.0, 1.0)
NOTCH = "#2f6fd0"
SPAWN = "#0c7ca4"
NARROW = "#d0312d"
PATH = "#e07b00"  # patrol paths (plan_data[PlanKey.PATROL_PATHS])


def _corners(p: Primitive) -> list[tuple[float, float]]:
    """Footprint corners of a primitive (a cylinder as its bounding square, a polygon prism as its vertices)."""
    if p.kind is Kind.POLYGON:
        return list(p.vertices)
    if p.kind is Kind.CYLINDER:
        x0, x1, y0, y1 = p.footprint_bounds()
        return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
    hx, hy = p.size[0] / 2, p.size[1] / 2
    c, s = math.cos(p.yaw), math.sin(p.yaw)
    return [
        (p.x + c * u - s * v, p.y + s * u + c * v)
        for u, v in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy))
    ]


def plan_png(
    world: World,
    a: Analysis,
    path: Path,
    width_px: int = 1600,
    crop: tuple[float, float, float, float] | None = None,
) -> None:
    """`review/plan.png` (T0 §8); with `crop` (xmin, xmax, ymin, ymax) the `plan_detail.png` view."""
    g = a.grid
    rgb = np.empty((g.ny, g.nx, 3))
    rgb[:] = OUTSIDE
    rgb[a.occupied] = SOLID
    rgb[a.reachable] = FREE
    rgb[a.sealed] = SEALED
    extent = (g.x0, g.x0 + g.nx * g.h, g.y0, g.y0 + g.ny * g.h)
    view = crop or extent
    span_x, span_y = view[1] - view[0], view[3] - view[2]
    dpi = 100
    fig_w = width_px / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * span_y / span_x + 0.9), dpi=dpi)
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest", zorder=0)
    for xv in range(math.ceil(view[0]), math.floor(view[1]) + 1):
        ax.axvline(
            xv, color="#b9b9b9" if xv % 5 else "#8a8a8a", lw=0.3 if xv % 5 else 0.6, zorder=1
        )
    for yv in range(math.ceil(view[2]), math.floor(view[3]) + 1):
        ax.axhline(
            yv, color="#b9b9b9" if yv % 5 else "#8a8a8a", lw=0.3 if yv % 5 else 0.6, zorder=1
        )
    step = 10 if max(span_x, span_y) > 60 else 5
    ax.set_xticks(range(step * math.ceil(view[0] / step), math.floor(view[1]) + 1, step))
    ax.set_yticks(range(step * math.ceil(view[2] / step), math.floor(view[3]) + 1, step))
    ax.tick_params(labelsize=9)
    for poly in a.inclusion_polygons:
        ax.add_patch(
            Polygon(poly, closed=True, fill=False, ec="#5b8f3a", lw=0.9, ls="--", zorder=3)
        )
    for p in world.primitives:
        if p.role is Role.LANDMARK:
            ax.add_patch(Polygon(_corners(p), closed=True, fc=NOTCH, ec=NOTCH, zorder=4))
    for n in world.notches:
        ax.add_patch(Polygon(n.polygon, closed=True, fc=NOTCH, ec=NOTCH, alpha=0.75, zorder=4))
        cx = sum(p[0] for p in n.polygon) / len(n.polygon)
        cy = sum(p[1] for p in n.polygon) / len(n.polygon)
        ax.annotate(
            n.id,
            (cx, cy),
            xytext=(0, 9),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            color=NOTCH,
        )
    for d in world.dimensions:
        if crop and not all(
            view[0] <= x <= view[1] and view[2] <= y <= view[3] for x, y in (d.p0, d.p1)
        ):
            continue  # a dimension outside the detail crop would stretch the axes
        ax.annotate(
            "",
            d.p1,
            d.p0,
            arrowprops={"arrowstyle": "<->", "color": "#333333", "lw": 1.0},
            zorder=5,
        )
        mx, my = (d.p0[0] + d.p1[0]) / 2, (d.p0[1] + d.p1[1]) / 2
        ax.text(
            mx,
            my,
            d.label,
            ha="center",
            va="bottom",
            fontsize=9,
            color="#222222",
            zorder=5,
            bbox={"fc": "white", "ec": "none", "alpha": 0.8, "pad": 1},
        )
    for route_id, route in (world.plan_data.get(PlanKey.PATROL_PATHS) or {}).items():
        polylines = route[PlanKey.POLYLINES_M]
        for line in polylines:
            xs, ys = zip(*line, strict=True)
            ax.plot(xs, ys, color=PATH, lw=1.4, ls="--", zorder=5)
        x0, y0 = polylines[0][0]
        ax.annotate(
            route_id, (x0, y0), xytext=(4, 4), textcoords="offset points", fontsize=8, color=PATH
        )
    arrow_len = max(0.8, 0.035 * max(span_x, span_y))
    for s in world.spawns:
        ax.add_patch(
            FancyArrow(
                s.x,
                s.y,
                arrow_len * math.cos(s.yaw),
                arrow_len * math.sin(s.yaw),
                width=arrow_len * 0.12,
                head_width=arrow_len * 0.4,
                head_length=arrow_len * 0.35,
                color=SPAWN,
                zorder=6,
            )
        )
        ax.plot([s.x], [s.y], "o", color=SPAWN, ms=5, zorder=7)
        ax.annotate(
            s.id,
            (s.x, s.y),
            xytext=(6, -12),
            textcoords="offset points",
            fontsize=10,
            color=SPAWN,
            weight="bold",
            zorder=7,
        )
    if a.narrowest_cell is not None and a.narrowest_m is not None:
        nx_, ny_ = g.centre(*a.narrowest_cell)
        ax.add_patch(Circle((nx_, ny_), a.narrowest_m / 2, fill=False, ec=NARROW, lw=1.6, zorder=8))
        ax.annotate(
            f"narrowest {a.narrowest_m:.2f} m",
            (nx_, ny_),
            xytext=(10, 10),
            textcoords="offset points",
            color=NARROW,
            fontsize=9,
            weight="bold",
            zorder=8,
        )
    bar = 5.0 if span_x < 80 else 10.0
    bx, by = view[0] + 0.03 * span_x, view[2] + 0.04 * span_y
    ax.plot([bx, bx + bar], [by, by], color="black", lw=3, zorder=9, solid_capstyle="butt")
    ax.text(bx + bar / 2, by + 0.012 * span_y, f"{bar:.0f} m", ha="center", fontsize=9, zorder=9)
    nx0, ny0 = view[1] - 0.04 * span_x, view[3] - 0.10 * span_y
    ax.annotate(
        "N",
        (nx0, ny0 + 0.06 * span_y),
        (nx0, ny0),
        ha="center",
        fontsize=11,
        weight="bold",
        arrowprops={"arrowstyle": "-|>", "color": "black", "lw": 1.4},
        zorder=9,
    )
    ax.set_xlim(view[0], view[1])
    ax.set_ylim(view[2], view[3])
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, east)")
    ax.set_ylabel("y (m, north)")
    ax.set_title(
        f"{world.id} {world.title}{'   detail' if crop else ''}   free {a.free_area_m2:.2f} m²   "
        f"worldgen {WORLDGEN_VERSION}",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def skeleton_png(world: World, a: Analysis, path: Path, width_px: int = 1600) -> None:
    """T1 review image: region map, skeleton, dead ends (red) and junctions (blue) over the plan."""
    from scipy import ndimage

    g = a.grid
    n_regions, grown = regions(a.reachable, a.clearance, g.h)
    palette = np.array(
        [
            (0.93, 0.80, 0.70),
            (0.75, 0.86, 0.95),
            (0.80, 0.92, 0.78),
            (0.95, 0.90, 0.70),
            (0.88, 0.80, 0.93),
            (0.78, 0.90, 0.90),
        ]
    )
    rgb = np.empty((g.ny, g.nx, 3))
    rgb[:] = OUTSIDE
    rgb[a.occupied] = SOLID
    rgb[a.reachable] = FREE
    for lab in range(1, int(grown.max()) + 1):
        rgb[grown == lab] = palette[(lab - 1) % len(palette)]
    sk = a.skeleton
    rgb[ndimage.binary_dilation(sk, structure=EIGHT)] = (0.15, 0.15, 0.15)
    end_cells, junction_cells = skeleton_nodes(sk)
    ends, joins = np.argwhere(end_cells), np.argwhere(junction_cells)
    extent = (g.x0, g.x0 + g.nx * g.h, g.y0, g.y0 + g.ny * g.h)
    span_x, span_y = extent[1] - extent[0], extent[3] - extent[2]
    dpi = 100
    fig_w = width_px / dpi
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * span_y / span_x + 0.9), dpi=dpi)
    ax.imshow(rgb, origin="lower", extent=extent, interpolation="nearest")
    if len(ends):
        xs, ys = g.x0 + (ends[:, 1] + 0.5) * g.h, g.y0 + (ends[:, 0] + 0.5) * g.h
        ax.plot(xs, ys, "o", color=NARROW, ms=7, label=f"dead ends ({len(ends)})")
    if len(joins):
        xs, ys = g.x0 + (joins[:, 1] + 0.5) * g.h, g.y0 + (joins[:, 0] + 0.5) * g.h
        ax.plot(xs, ys, "s", color="#2f6fd0", ms=5, label="junction cells")
    if len(ends) or len(joins):
        ax.legend(loc="upper left", fontsize=9)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m, east)")
    ax.set_ylabel("y (m, north)")
    ax.set_title(
        f"{world.id} {world.title}: skeleton and {n_regions} regions ({DESCRIPTORS_VERSION})",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def family_sheet(
    family: str,
    entries: list[tuple[World, Analysis, dict[str, Any]]],
    columns: int,
    path: Path,
) -> None:
    """Every instance's plan side by side at one metric scale (T0 §8), labelled with id, seed and the family's
    `plan_data.sheet_label` (default: loops and dead ends, the generator's count when it records one, else
    T1's skeleton count)."""
    half = max(max(abs(v) for v in world.bounds()) for world, _, _ in entries) + 1.0
    rows = math.ceil(len(entries) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(4.2 * columns, 4.5 * rows), squeeze=False)
    for ax in axes.flat:
        ax.set_axis_off()
    for k, (world, a, data) in enumerate(entries):
        ax = axes.flat[k]
        g = a.grid
        extent = (g.x0, g.x0 + g.nx * g.h, g.y0, g.y0 + g.ny * g.h)
        img = np.where(a.occupied, 0.25, np.where(a.reachable, 0.95, 1.0))
        ax.imshow(img, origin="lower", cmap="gray", vmin=0, vmax=1, extent=extent)
        for p in world.primitives:
            if p.role is Role.LANDMARK:
                ax.add_patch(Polygon(_corners(p), closed=True, fc=NOTCH, ec=NOTCH, zorder=3))
        for s in world.spawns:
            ax.add_patch(
                FancyArrow(
                    s.x,
                    s.y,
                    1.5 * math.cos(s.yaw),
                    1.5 * math.sin(s.yaw),
                    width=0.25,
                    color=SPAWN,
                    length_includes_head=True,
                )
            )
        extra = data.get("plan_data") or {}
        dead = extra.get(PlanKey.DEAD_END_CELLS, data["descriptors"]["dead_ends"])
        label = extra.get(
            PlanKey.SHEET_LABEL, f"loops {data['measured']['loops']}, dead ends {dead}"
        )
        ax.set_title(f"{world.id}  seed {data['seed']}\n{label}", fontsize=10)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_axis_on()
        ax.tick_params(labelsize=7)
    fig.suptitle(f"{family} family, one metric scale (axes in m)   worldgen {WORLDGEN_VERSION}")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
