"""Render the manuscript figures from archived data only.

Run: python plotting/paper_figures.py --figures 7 8 9 10 11 12 14 15 16
PNG output is at least 801 dpi. Individual panels are suitable for layout tools.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle, ConnectionPatch
from matplotlib.ticker import MaxNLocator
import numpy as np

from plot_data import (
    ROOT, SENSITIVE, METHODS, PARAMETERS, INPUTS, AUDIT,
    load_reward, ema, source, site01_data, stability_data,
)


# Preserve five original method colors. Give PPO a rose color distinct from
# PPO+GRU blue, and replace the formerly pale SAC with a stronger hue.
METHOD_STYLE = {
    "PPO+GRU": dict(color="#1F77B4", linestyle="-", marker="o", linewidth=1.15),
    "PPO": dict(color="#D33682", linestyle=(0, (6, 2)), marker="s", linewidth=.9),
    "SAC+GRU": dict(color="#2CA02C", linestyle="-", marker="^", linewidth=.9),
    "SAC": dict(color="#8C564B", linestyle=(0, (6, 2)), marker="v", linewidth=.9),
    "DQN": dict(color="#9467BD", linestyle="-", marker="D", linewidth=.9),
    "Astar_shadow": dict(color="#FF7F0E", linestyle=(0, (9, 2.5)), marker="", linewidth=.9),
    "Astar": dict(color="#D62728", linestyle="-", marker="x", linewidth=.9),
}
# Retain the original red / green / blue sequence, with a less pale green.
PARAM_STYLE = [
    dict(color="#C44E52", linestyle="-", marker="s", linewidth=.9),
    dict(color="#2F944D", linestyle="-", marker="o", linewidth=.9),
    dict(color="#3F70B5", linestyle="-", marker="^", linewidth=.9),
]
PARAM_TITLES = {
    "lr": "Reward curve for different learning rate",
    "gamma": "Reward curve for different discount factor gamma",
    "lambda": "Reward curve for different weight lambda",
    "clip": "Reward curve for different clipping ratio",
    "entropy coeff": "Reward curve for different entropy coefficient",
}
STATES = [
    ("d", "Illumination", "Illumination", "Illumination along the path", None, None),
    ("e", "Slope", "Slope", "Slope change along the path", (0, 30), (0, 18)),
    ("f", "Thermal", "Thermal", "Thermal change along the path", None, (-20, 40)),
    ("g", "Battery", "Battery", "Battery change along the path", (0, 100), (30, 100)),
]
OUTPUTS = []
EXPORT_DPI_OVERRIDES = json.loads((Path(__file__).parent/"reviewer_export_dpi.json").read_text(encoding="utf-8")) if (Path(__file__).parent/"reviewer_export_dpi.json").exists() else {}
STD_MULTIPLIER = .95
MARKER_SIZE = 4.0
MARKER_EDGE_WIDTH = .7
MEAN_LINE_WIDTH = 1.1
RAW_REWARD_LINE_WIDTH = .35
METHOD_REWARD_EMA_ALPHA = .015
# Reward curves can carry a slightly stronger stroke after smoothing, while
# the path and state panels retain their selected stroke widths.
METHOD_REWARD_STYLE = {
    method: dict(METHOD_STYLE[method], linewidth=1.3 if method == "PPO+GRU" else 1.05)
    for method in METHODS[:5]
}
METHOD_PATH_STYLE = {
    method: dict(METHOD_STYLE[method], linewidth=1.65 if method == "PPO+GRU" else 1.4)
    for method in METHODS
}
# Focus on a moderate interval where the training curves overlap. Each interval
# lies within the common recorded duration; the main axes retain all data.
ZOOM_SPECS = {
    "fig07/lr": {"xlim": (1800, 3300)},
    "fig07/gamma": {"xlim": (2800, 4400)},
    "fig07/lambda": {"xlim": (2400, 4000)},
    "fig07/clip": {"xlim": (1300, 2700)},
    "fig07/entropy coeff": {"xlim": (2400, 4000)},
    "fig09/reward": {"xlim": (2800, 4400), "line_widths": (1.1, .9, .9, .9, .9)},
    "fig10/reward": {"xlim": (1300, 2300), "line_widths": (1.1, .9, .9, .9, .9)},
    "fig11/reward": {"xlim": (2800, 4400), "line_widths": (1.1, .9, .9, .9, .9)},
    "fig14/reward": {"xlim": (1200, 2100), "line_widths": (1.1, .9)},
}
ZOOM_MARKER_SIZE = 3.1
ZOOM_MARKER_EDGE_WIDTH = .55


def configure():
    # Match the original code: default sans-serif, full rectangular axes,
    # centered titles, no grid, white background. Only curve styling changes.
    plt.rcdefaults()
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42,
                         "path.simplify": False,
                         # Dashes above are in points. Keep the previous dash
                         # spacing while changing stroke thickness alone.
                         "lines.scale_dashes": False})


def save(fig, stem, out, dpi):
    path = out / stem
    path.parent.mkdir(parents=True, exist_ok=True)
    # Preserve the figure geometry; render additional pixels where Visio uses
    # a larger physical rectangle. 801 also avoids PNG pHYs rounding below 800.
    export_dpi = max(801, dpi, EXPORT_DPI_OVERRIDES.get(stem, 801))
    for ext in ("svg", "pdf", "png"):
        target = path.with_suffix("." + ext)
        fig.savefig(target, dpi=export_dpi)
        OUTPUTS.append(target.relative_to(out).as_posix())
    preview = out / "previews" / (path.name + ".png")
    fig.savefig(preview, dpi=140 if "panels/" in stem else 150)
    plt.close(fig)
    print(f"Exported {stem}", flush=True)


def common_marker_interval(series, target_count=11):
    """Use one x-step interval for every curve in the same axes."""
    return max(1, int(np.ceil((max(map(len, series))-1) / target_count)))


def marker_every(index, interval, group_size):
    # Stagger symbols between curves, keeping every phase within one period.
    return (index * max(1, interval // group_size)) % interval, interval


def foreground_zorder(style):
    """Keep PPO+GRU above comparison data, below annotations and inset axes."""
    return 4 if style["color"].lower() == METHOD_STYLE["PPO+GRU"]["color"].lower() else 2


def path_with_stay_markers(ax, path, style, phase, interval, path_effects=None):
    """Keep stay dots visible on paths, beneath opaque hollow method markers."""
    zorder = foreground_zorder(style)
    line_style = dict(style, marker="")
    line, = ax.plot(path[:, 1]-1, path[:, 0]-1, **line_style, zorder=zorder)
    if path_effects:
        line.set_path_effects(path_effects)
    if style.get("marker"):
        stays = np.flatnonzero(np.all(np.diff(path, axis=0)==0, axis=1))+1
        if len(stays):
            color = np.array(matplotlib.colors.to_rgb(style["color"]))* .7
            ax.scatter(path[stays, 1]-1, path[stays, 0]-1, s=.1, marker="s",
                       color=color, zorder=zorder+.1)
        # Every sparse method marker is above every stay-dot layer. Preserve
        # PPO+GRU priority within this layer, with endpoints still above both.
        markers, = ax.plot(path[:, 1]-1, path[:, 0]-1, color=style["color"],
                           linestyle="None", marker=style["marker"], markersize=3.7,
                           markerfacecolor="white", markeredgewidth=MARKER_EDGE_WIDTH,
                           markevery=(phase, interval),
                           zorder=5.2 if zorder == 4 else 5)
        if path_effects:
            markers.set_path_effects(path_effects)
    return line


def curve(ax, values, style, index=0, x=None, marker_count=11,
          marker_interval=None, marker_group_size=5,
          marker_size=MARKER_SIZE, marker_edge_width=MARKER_EDGE_WIDTH, **kwargs):
    interval = (common_marker_interval([values], marker_count)
                if marker_interval is None else marker_interval)
    return ax.plot(np.arange(len(values)) if x is None else x, values, **style,
                   markersize=marker_size, markerfacecolor="white", markeredgewidth=marker_edge_width,
                   markevery=marker_every(index, interval, marker_group_size),
                   **kwargs)[0]


def zoom(ax, series, styles, key):
    start, end = ZOOM_SPECS[key]["xlim"]
    if not 0 <= start < end < min(map(len, series)):
        raise ValueError(f"Zoom interval is outside the common recorded data: {key}")
    local_series = [values[start:end+1] for values in series]
    low = min(float(v.min()) for v in local_series)
    high = max(float(v.max()) for v in local_series)
    margin = max(10, (high-low) * .08)
    main_xlim, main_ylim = ax.get_xlim(), ax.get_ylim()
    # A smaller inset stays below the selected curves and above the legend.
    top = min(.66, (low-margin-main_ylim[0])/(main_ylim[1]-main_ylim[0])-.08)
    bottom = .34
    inset = ax.inset_axes([.40, bottom, .55, top-bottom])
    # Inset stroke widths are independent of the main-axes widths.
    widths = ZOOM_SPECS[key].get("line_widths", [.8]*len(series))
    marker_interval = common_marker_interval(local_series, 7)
    for i, (values, style) in enumerate(zip(local_series, styles)):
        local_style = dict(style, linewidth=widths[i])
        curve(inset, values, local_style, i, x=np.arange(start, end+1),
              marker_count=7, marker_size=ZOOM_MARKER_SIZE,
              marker_interval=marker_interval, marker_group_size=len(series),
              marker_edge_width=ZOOM_MARKER_EDGE_WIDTH, zorder=foreground_zorder(style))
    inset.set_xlim(start, end)
    inset.set_ylim(low-margin, high+margin)
    inset.set_title("Zoomed view", fontsize=9, pad=3)
    inset.tick_params(labelsize=8, length=3)
    inset.xaxis.set_major_locator(MaxNLocator(3))
    inset.yaxis.set_major_locator(MaxNLocator(3))
    for spine in inset.spines.values():
        spine.set_linewidth(.7)
    # Identify exactly the range shown in the inset. Two restrained leaders
    # join its upper corners to the lower corners of that range in the main
    # axes, i.e. the beginning and end of the zoomed episode interval.
    zoom_xlim, zoom_ylim = inset.get_xlim(), inset.get_ylim()
    ax.add_patch(Rectangle((zoom_xlim[0], zoom_ylim[0]),
                           zoom_xlim[1]-zoom_xlim[0], zoom_ylim[1]-zoom_ylim[0],
                           fill=False, edgecolor="#777777", linewidth=.65,
                           alpha=.8, zorder=1.4))
    for side, episode in ((0, zoom_xlim[0]), (1, zoom_xlim[1])):
        connection = ConnectionPatch(xyA=(side, 1), xyB=(episode, zoom_ylim[0]),
                                     coordsA="axes fraction", coordsB="data",
                                     axesA=inset, axesB=ax, color="#777777",
                                     linewidth=.65, alpha=.8, zorder=1.3,
                                     clip_on=False)
        ax.add_artist(connection)
    ax.set_xlim(main_xlim)
    ax.set_ylim(main_ylim)
    AUDIT[f"{key}/zoom"] = dict(xlim=list(zoom_xlim), ylim=list(zoom_ylim),
                                line_widths=widths, marker_size=ZOOM_MARKER_SIZE,
                                marker_interval=marker_interval,
                                marker_edge_width=ZOOM_MARKER_EDGE_WIDTH,
                                source="same EMA values as main axes, no resampling")
    return inset


def sublabel(ax, letter, lower=-.22):
    ax.text(.5, lower, f"({letter})", transform=ax.transAxes, ha="center",
            va="top", fontfamily="Times New Roman", fontsize=18)


def rewards(ax, series, styles, names, title, legend=True, fonts=False, zoom_key=None,
            ema_alpha=.05):
    # Retain every raw value. The same EMA (adjust=True) feeds the main and
    # inset axes; its coefficient is recorded for each reward comparison.
    for values, style in zip(series, styles):
        ax.plot(values, color=style["color"], alpha=.08, linewidth=RAW_REWARD_LINE_WIDTH, zorder=1)
    smoothed = [ema(values, ema_alpha) for values in series]
    marker_interval = common_marker_interval(smoothed)
    marker_counts = {}
    for i, (values, style, name) in enumerate(zip(smoothed, styles, names)):
        line = curve(ax, values, style, i, marker_interval=marker_interval,
                     marker_group_size=len(series), label=name, zorder=foreground_zorder(style))
        phase, interval = line.get_markevery()
        marker_counts[name] = len(range(phase, len(values), interval)) if style.get("marker") else 0
    if zoom_key is not None:
        AUDIT[f"{zoom_key}/markers"] = dict(interval=marker_interval,
            counts=marker_counts, rule="common x-step interval based on the longest displayed curve")
    ax.set_xlabel("Episode", fontsize=12 if fonts else 10)
    ax.set_ylabel("Reward", fontsize=12 if fonts else 10)
    ax.set_title(title, fontsize=15 if fonts else 12)
    if zoom_key is not None:
        zoom(ax, smoothed, styles, zoom_key)
    if legend:
        ax.legend(loc="lower right", handlelength=3.8, frameon=True)


def figure7(out, dpi):
    rows = []
    for letter, folder, title, vals, _ in PARAMETERS:
        series = []
        names = []
        for value in vals:
            values = load_reward(SENSITIVE/folder/value/"reward.csv")
            original_n = len(values)
            if folder in ("lambda", "entropy coeff") and len(values) > 6000:
                values = values[:5100]
            series.append(values)
            names.append(("learning_rate" if folder == "lr" else folder) + "_" + value)
            AUDIT[f"fig07/{folder}/{value}"] = dict(source_episodes=original_n,
                displayed_episodes=len(values), ema_alpha=.05)
        rows.append((letter, folder, series, names))
    fig = plt.figure(figsize=(12.8, 15.4))
    grid = fig.add_gridspec(3, 4, left=.075, right=.985, top=.96, bottom=.07,
                           wspace=1.1, hspace=.56)
    positions = [(0, slice(0, 2)), (0, slice(2, 4)), (1, slice(0, 2)),
                 (1, slice(2, 4)), (2, slice(1, 3))]
    for position, (letter, folder, series, names) in zip(positions, rows):
        ax = fig.add_subplot(grid[position[0], position[1]])
        rewards(ax, series, PARAM_STYLE, names, PARAM_TITLES[folder], zoom_key=f"fig07/{folder}")
        sublabel(ax, letter)
    save(fig, "fig07_original_style", out, dpi)
    for letter, folder, series, names in rows:
        fig, ax = plt.subplots()
        rewards(ax, series, PARAM_STYLE, names, PARAM_TITLES[folder], zoom_key=f"fig07/{folder}")
        save(fig, f"panels/fig07_{letter}_{folder.replace(' ', '_')}", out, dpi)


def stability(ax, mean, sd, site, legend=True):
    style = dict(color="blue", linestyle="-", linewidth=MEAN_LINE_WIDTH)
    ax.plot(mean, label="Mean Reward", **style)
    ax.fill_between(np.arange(len(mean)), mean-STD_MULTIPLIER*sd,
                    mean+STD_MULTIPLIER*sd, color="blue", alpha=.2,
                    label="Standard Deviation", linewidth=0)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title(f"{site} scenarios Reward Mean and Standard Deviation")
    ax.set_ylim(-1600, 250)
    ax.set_yticks(np.arange(-1500, 301, 250))
    if legend:
        ax.legend(loc="lower right")


def figure8(out, dpi):
    rows = [(letter, site, *stability_data(site.lower()))
            for letter, site in zip("abc", ("Site01", "Site06", "Site07"))]
    fig = plt.figure(figsize=(12.8, 10.2))
    grid = fig.add_gridspec(2, 4, left=.075, right=.985, top=.94, bottom=.11,
                           wspace=1.1, hspace=.55)
    positions = [(0, slice(0, 2)), (0, slice(2, 4)), (1, slice(1, 3))]
    for position, (letter, site, mean, sd) in zip(positions, rows):
        ax = fig.add_subplot(grid[position[0], position[1]])
        stability(ax, mean, sd, site, False)
        sublabel(ax, letter)
    handles = [Line2D([], [], color="blue", linewidth=MEAN_LINE_WIDTH, label="Mean Reward"),
               Patch(color="blue", alpha=.2, label="Standard Deviation")]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.77, .43), fontsize=12)
    save(fig, "fig08_original_style", out, dpi)
    for letter, site, mean, sd in rows:
        fig, ax = plt.subplots()
        stability(ax, mean, sd, site)
        save(fig, f"panels/fig08_{letter}_{site.lower()}", out, dpi)
    fig, ax = plt.subplots(figsize=(4.4, 1.1))
    ax.legend(handles=handles, loc="center", fontsize=14)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "panels/fig08_standard_deviation_legend", out, dpi)


def method_handles(methods=None, styles=None):
    methods = METHODS if methods is None else methods
    styles = METHOD_STYLE if styles is None else styles
    return [Line2D([], [], **styles[m], markersize=MARKER_SIZE, markerfacecolor="white",
                   markeredgewidth=MARKER_EDGE_WIDTH, label=m) for m in
            ("Astar", "Astar_shadow", "DQN", "PPO", "PPO+GRU", "SAC", "SAC+GRU") if m in methods]


def path_panel(ax, data, site="site01", audit_key="fig09"):
    slope = np.loadtxt(source(ROOT/site/"slope.csv"), delimiter=",")
    gray = np.clip(slope / 18 * 255, 0, 255)
    gray[slope >= 18] = 0
    ax.imshow(np.repeat(gray.astype(np.uint8)[..., None], 3, axis=2))
    methods = list(data)
    interval = common_marker_interval([data[m]["path"] for m in methods], 10)
    marker_counts = {}
    for i, method in enumerate(methods):
        p = data[method]["path"]
        style = METHOD_PATH_STYLE[method]
        phase, _ = marker_every(i, interval, len(methods))
        marker_counts[method] = len(range(phase, len(p), interval)) if style.get("marker") else 0
        path_with_stay_markers(ax, p, style, phase, interval)
    reference_path = data["PPO+GRU"]["path"]
    endpoints = {}
    for label, point in (("Start", reference_path[0]), ("Goal", reference_path[-1])):
        offset, alignment = ((0, -14), "top") if point[0] > 100 else ((0, 14), "bottom")
        xy = (point[1]-1, point[0]-1)
        ax.scatter(*xy, s=28, facecolors="white", edgecolors="black",
                   linewidths=.9, zorder=6)
        ax.annotate(label, xy=xy, xytext=offset, textcoords="offset points",
                    ha="center", va=alignment, fontsize=11, color="black",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1.5),
                    zorder=7)
        endpoints[label] = dict(grid_row_column=point.tolist(), displayed_xy=[int(v) for v in xy])
    ax.axis("off")
    AUDIT[f"{audit_key}/path/endpoints"] = endpoints
    AUDIT[f"{audit_key}/path/markers"] = dict(interval=interval, counts=marker_counts,
        rule="common recorded-step interval based on the longest path")


def gantt_panel(ax, data):
    gap, height = .15, .075
    methods = list(data)
    for i, method in enumerate(methods):
        # Retain the original per-method colored move/stay strips, including
        # the initial path point and its move-colored cell.
        stay = np.r_[False, data[method]["actions"]==0]
        breaks = np.r_[0, np.flatnonzero(np.diff(stay.astype(int)))+1, len(stay)]
        color = np.array(matplotlib.colors.to_rgb(METHOD_STYLE[method]["color"]))
        for start, end in zip(breaks[:-1], breaks[1:]):
            ax.broken_barh([(start-.5, end-start)], (i*gap-height/2, height),
                           facecolors=color*(.7 if stay[start] else 1), alpha=.9)
    ax.set_yticks(np.arange(len(methods))*gap, methods)
    ax.set_xlabel("Step")
    ax.set_title("Stay/Move Action Gantt Chart", fontsize=12)
    ax.set_xlim(-.5, max(len(d["path"]) for d in data.values())+20)
    ax.set_ylim((len(methods)-1)*gap+height, -height)


def state_panel(ax, data, spec, audit_key="fig09"):
    letter, key, ylabel, title, limits, band = spec
    methods = list(data)
    marker_interval = common_marker_interval([data[m][key] for m in methods])
    marker_counts = {}
    for i, method in enumerate(methods):
        line = curve(ax, data[method][key], METHOD_STYLE[method], i,
              marker_interval=marker_interval, marker_group_size=len(methods),
              drawstyle="default" if key=="Illumination" else "steps-post",
              zorder=foreground_zorder(METHOD_STYLE[method]))
        phase, interval = line.get_markevery()
        marker_counts[method] = len(range(phase, len(data[method][key]), interval)) if METHOD_STYLE[method].get("marker") else 0
    AUDIT[f"{audit_key}/{key.lower()}/markers"] = dict(interval=marker_interval,
        counts=marker_counts, rule="common x-step interval based on the longest displayed curve")
    if band:
        ax.fill_between(np.arange(max(len(d["path"]) for d in data.values())),
                        *band, facecolor="lime", alpha=.15, zorder=0)
    if limits:
        ax.set_ylim(*limits)
    ax.set_xlabel("Step", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=16)


def figure9(out, dpi):
    print("Loading the verified manuscript Site01 data...", flush=True)
    data = site01_data()
    learners = METHODS[:5]
    series = []
    for method in learners:
        values = load_reward(ROOT/f"tmp_{method}/site01/reward.csv")
        original_n = len(values)
        if len(values)>7000:
            values = values[:6100]
        series.append(values)
        AUDIT[f"fig09/rewards/{method}"] = dict(source_episodes=original_n,
            displayed_episodes=len(values), ema_alpha=METHOD_REWARD_EMA_ALPHA)
    styles = [METHOD_REWARD_STYLE[m] for m in learners]
    fig, axes = plt.subplots(4, 2, figsize=(12.8, 20.4))
    fig.subplots_adjust(left=.085, right=.985, top=.97, bottom=.07, wspace=.26, hspace=.57)
    rewards(axes[0, 0], series, styles, learners, "Reward curve", False, True,
            zoom_key="fig09/reward", ema_alpha=METHOD_REWARD_EMA_ALPHA)
    path_panel(axes[0, 1], data)
    gantt_panel(axes[1, 0], data)
    for ax, spec in zip([axes[1, 1], axes[2, 0], axes[2, 1], axes[3, 0]], STATES):
        state_panel(ax, data, spec)
    for ax, letter in zip(axes.flat, "abcdefg"):
        sublabel(ax, letter)
    axes[3, 1].legend(handles=method_handles(), loc="upper center", handlelength=4.3,
                      labelspacing=.65, fontsize=14, frameon=True)
    axes[3, 1].axis("off")
    save(fig, "fig09_original_style", out, dpi)
    fig, ax = plt.subplots()
    rewards(ax, series, styles, learners, "Reward curve", False, True,
            zoom_key="fig09/reward", ema_alpha=METHOD_REWARD_EMA_ALPHA)
    save(fig, "panels/fig09_a_reward", out, dpi)
    fig, ax = plt.subplots()
    path_panel(ax, data)
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.98)
    save(fig, "panels/fig09_b_path", out, dpi)
    fig, ax = plt.subplots(figsize=(4, 3))
    gantt_panel(ax, data)
    fig.tight_layout()
    save(fig, "panels/fig09_c_actions", out, dpi)
    for spec in STATES:
        fig, ax = plt.subplots()
        state_panel(ax, data, spec)
        save(fig, f"panels/fig09_{spec[0]}_{spec[1].lower()}", out, dpi)
    fig, ax = plt.subplots(figsize=(2.8, 4.3))
    ax.legend(handles=method_handles(), loc="center", fontsize=14, handlelength=4.3,
              labelspacing=.65, frameon=True)
    ax.axis("off")
    fig.tight_layout()
    save(fig, "panels/fig09_methods_legend", out, dpi)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    all_figures = (7, 8, 9, 10, 11, 12, 14, 15, 16)
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/paper_figures")
    parser.add_argument("--dpi", type=int, default=800)
    parser.add_argument("--figures", type=int, nargs="+", choices=all_figures, default=list(all_figures))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/"previews").mkdir(exist_ok=True)
    existing = args.output/"data_audit.json"
    if existing.exists() and set(args.figures)!=set(all_figures):
        previous = json.loads(existing.read_text(encoding="utf-8"))
        INPUTS.update(previous["input_files"])
        AUDIT.update(previous["checks_and_display_settings"])
        OUTPUTS.extend(previous["output_files"])
    configure()
    for number, render in ((7, figure7), (8, figure8), (9, figure9)):
        if number in args.figures:
            render(args.output, args.dpi)
    if any(number > 9 for number in args.figures):
        from trajectory_figures import render_remaining
        render_remaining(sys.modules[__name__], args.output, args.dpi, args.figures)
    OUTPUTS[:] = sorted(set(OUTPUTS))
    for relative, details in INPUTS.items():
        digest = hashlib.sha256()
        with (ROOT/relative).open("rb") as stream:
            for block in iter(lambda: stream.read(1024*1024), b""):
                digest.update(block)
        if digest.hexdigest()!=details["sha256"]:
            raise RuntimeError(f"Source changed while rendering: {relative}")
    batch_figures = sorted({int(name[3:5]) for name in OUTPUTS if name.startswith("fig")})
    manifest = dict(script=Path(__file__).name, figures=batch_figures, input_files=INPUTS,
                    checks_and_display_settings=AUDIT, method_styles=METHOD_STYLE,
                    parameter_styles=PARAM_STYLE, marker_size=MARKER_SIZE,
                    marker_edge_width=MARKER_EDGE_WIDTH, dash_units="points, unscaled",
                    marker_placement="common interval per axes from the longest displayed curve; staggered phases; separate interval for each inset",
                    zoom_connectors="two leaders from inset upper corners to highlighted range lower corners",
                    zoom_specs=ZOOM_SPECS, figure8_zoom=False,
                    raw_reward_alpha=.08, raw_reward_line_width=RAW_REWARD_LINE_WIDTH,
                    figure8_mean_line_width=MEAN_LINE_WIDTH, std_band_multiplier=STD_MULTIPLIER,
                    reward_smoothing={"fig07": {"alpha": .05, "adjust": True},
                                      **{f"fig{n:02d}": {"alpha": METHOD_REWARD_EMA_ALPHA, "adjust": True}
                                         for n in (9, 10, 11, 14) if n in batch_figures}},
                    figure9_reward_styles=METHOD_REWARD_STYLE,
                    figure9_path_styles=METHOD_PATH_STYLE,
                    method_foreground="PPO+GRU data zorder=4; other methods=2; endpoints and labels above data",
                    path_marker_layers="path lines=2/4; stay dots=2.1/4.1; sparse markers=5/5.2 (PPO+GRU higher); endpoints=6, labels=7",
                    minimum_png_export_dpi=801, export_dpi_overrides=EXPORT_DPI_OVERRIDES,
                    output_files=OUTPUTS)
    (args.output/"data_audit.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Verified unchanged sources; exported {len(OUTPUTS)} files.", flush=True)


if __name__ == "__main__":
    main()
