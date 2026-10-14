"""Path and state comparisons from archived manuscript-aligned records."""
from __future__ import annotations

import csv
import re
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import PowerNorm, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np

from plot_data import DIRECTIONS


SCENARIOS = {10: ("site06", 0), 11: ("site07", 0), 14: ("site01 660", 660)}
DATA_CACHE = {}
FRAME_CACHE = {}
TIME_TRAJECTORY_CACHE = {}
# A single path over a temperature map needs more weight than the comparison plots.
FIG16_PATH_LINEWIDTH = 2.2
FIG12_PATH_LINEWIDTHS = {"PPO+GRU": 2.2, "others": 1.8}


def table_time_trajectories(b, start_time):
    """Use the same recorded runs as Table 6 and Figs. 9/14.

    T=0 uses the 497-point PPO+GRU record (496 actions plus the start),
    reward 282.39. T=660 uses the 1381-point record, reward 286.04.
    All three methods use the runs listed in the manuscript tables.
    """
    if start_time in TIME_TRAJECTORY_CACHE:
        return TIME_TRAJECTORY_CACHE[start_time]
    data = {}
    site = "site01" if start_time == 0 else "site01 660"
    summary_site = "site01" if start_time == 0 else "site01_t660"
    summary_path = b.source(b.ROOT/f"results/tables/{summary_site}/result_info.csv")
    with summary_path.open(newline="") as stream:
        summary = {r["Algorithm"]: r for r in csv.DictReader(stream)}
    for method in ("PPO+GRU", "PPO", "Astar_shadow"):
        path = b.source(b.ROOT/f"tmp_{method}/{site}/result_info.csv")
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        actions = np.array([int(r["action"]) for r in rows])
        positions = np.array([[int(v) for v in re.findall(r"-?\d+", r["Position"])] for r in rows])
        initial = np.array([175, 25])
        reconstructed = initial+np.cumsum(DIRECTIONS[actions], axis=0)
        if not np.array_equal(positions, reconstructed) or positions[-1].tolist() != [25, 175]:
            raise ValueError(f"Original time-comparison trajectory is inconsistent: {method}, {start_time}")
        fullpath = np.vstack((initial, positions))
        reward = round(sum(float(r["Reward"]) for r in rows), 2)
        reference = summary[method]
        if (len(fullpath) != int(reference["Path length"])
                or reward != float(reference["Total reward"])
                or int((actions == 0).sum()) != int(reference["Stay actions"])):
            raise ValueError(f"Time-comparison data disagrees with Table 6: {method}, {start_time}")
        data[method] = dict(path=fullpath, actions=actions)
        b.AUDIT.pop(f"original_time_comparison/{method}-{start_time}", None)
        b.AUDIT[f"table6_time_comparison/{method}-{start_time}"] = dict(
            source=path.relative_to(b.ROOT).as_posix(), start_time=start_time,
            path_points_including_start=len(fullpath), recorded_actions=len(actions),
            total_reward=reward, matched_table6_summary=True,
            matched_recorded_positions=True,
            source_choice="same archived runs as Table 6 and Figs.9/14")
    TIME_TRAJECTORY_CACHE[start_time] = data
    return data


def scenario_data(b, site, figure, offset=0):
    key = (site, offset)
    if key in DATA_CACHE:
        return DATA_CACHE[key]
    print(f"Verifying manuscript results and illumination: {site}", flush=True)
    methods = b.METHODS if offset == 0 else ["PPO+GRU", "PPO", "Astar_shadow"]
    summary_site = "site01" if offset == 0 else "site01_t660"
    summary_path = b.source(b.ROOT/f"results/tables/{summary_site}/result_info.csv")
    summary = {r["Algorithm"]: r for r in csv.DictReader(summary_path.open())}
    data = {}
    for method in methods:
        path = b.source(b.ROOT/f"tmp_{method}/{site}/result_info.csv")
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        actions = np.array([int(r["action"]) for r in rows])
        positions = np.array([[int(v) for v in re.findall(r"-?\d+", r["Position"])] for r in rows])
        initial = positions[0]-DIRECTIONS[actions[0]]
        reconstructed = initial+np.cumsum(DIRECTIONS[actions], axis=0)
        goals = np.array([[int(v) for v in re.findall(r"-?\d+", r["Goal Position"])] for r in rows])
        if not np.array_equal(positions, reconstructed) or not np.all(goals == positions[-1]):
            raise ValueError(f"Recorded path reconstruction/goal mismatch: {site}, {method}")
        values = {name: np.array([float(r[name]) for r in rows])
                  for name in ("Reward", "Slope", "Thermal", "Battery")}
        if not all(np.all(np.isfinite(v)) for v in values.values()):
            raise ValueError(f"Nonfinite states: {site}, {method}")
        fullpath = np.vstack((initial, positions))
        reward = round(float(values["Reward"].sum()), 2)
        reference = summary[method]
        if len(fullpath) != int(reference["Path length"]) or reward != float(reference["Total reward"]):
            raise ValueError(f"Result disagrees with manuscript summary: {site}, {method}")
        if int((actions == 0).sum()) != int(reference["Stay actions"]):
            raise ValueError(f"Stay count disagrees with manuscript: {site}, {method}")
        for name, column in (("Slope", "Range slope"), ("Thermal", "Range thermal"), ("Battery", "Range battery")):
            limits = np.array([float(v) for v in re.findall(r"-?\d+(?:\.\d+)?", reference[column])])
            if not np.allclose([values[name].min(), values[name].max()], limits, atol=.0051):
                raise ValueError(f"State range disagrees with manuscript: {site}, {method}, {name}")
        data[method] = dict(values, actions=actions, path=fullpath)
        b.AUDIT[f"fig{figure:02d}/{method}"] = dict(
            site=site, start_time=offset, path_points_including_start=len(fullpath),
            recorded_actions=len(actions), total_reward=reward,
            start=initial.tolist(), goal=positions[-1].tolist(), matched_manuscript_summary=True)
    queries = {}
    for method, item in data.items():
        item["Illumination"] = np.full(len(item["path"]), np.nan)
        for step, (row, col) in enumerate(item["path"]):
            if not (1 <= row <= 200 and 1 <= col <= 200):
                raise ValueError(f"Path is outside the map: {site}, {method}, {step}")
            index = (int(col)-1)*1440+(step+offset) % 1440
            queries.setdefault(int(row)-1, []).append((method, step, index))
    terrain_site = site.split()[0]
    light = b.source(b.ROOT/terrain_site/"lunarSolarLight.csv")
    with light.open() as stream:
        for row, line in enumerate(stream):
            if row in queries:
                values = np.fromstring(line, sep=",")
                if len(values) != 200*1440:
                    raise ValueError(f"Unexpected temporal map layout: {light}, row {row}")
                for method, step, index in queries[row]:
                    data[method]["Illumination"][step] = values[index]
    if any(not np.all(np.isfinite(d["Illumination"])) for d in data.values()):
        raise ValueError(f"Missing illumination values: {site}")
    b.AUDIT[f"fig{figure:02d}/illumination_sampling"] = dict(
        source=f"{terrain_site}/lunarSolarLight.csv", layout="row, column * 1440 + time",
        index="path row/column minus one, (recorded step + start time) modulo 1440",
        start_time=offset, state_values="recorded CSV values, unchanged")
    DATA_CACHE[key] = data
    return data


def comparison(b, out, dpi, number, site, offset):
    data = scenario_data(b, site, number, offset)
    methods = list(data)
    learners = [m for m in methods if m in b.METHOD_REWARD_STYLE]
    series = []
    for method in learners:
        values = b.load_reward(b.ROOT/f"tmp_{method}/{site}/reward.csv")
        original_n = len(values)
        if len(values) > 7000:
            values = values[:6100]
        series.append(values)
        b.AUDIT[f"fig{number:02d}/rewards/{method}"] = dict(
            source_episodes=original_n, displayed_episodes=len(values), ema_alpha=b.METHOD_REWARD_EMA_ALPHA)
    styles = [b.METHOD_REWARD_STYLE[m] for m in learners]
    audit_key = f"fig{number:02d}"

    def reward(ax):
        b.rewards(ax, series, styles, learners, "Reward curve", False, True,
                  zoom_key=audit_key+"/reward", ema_alpha=b.METHOD_REWARD_EMA_ALPHA)

    fig, axes = plt.subplots(4, 2, figsize=(12.8, 20.4))
    fig.subplots_adjust(left=.085, right=.985, top=.97, bottom=.07, wspace=.26, hspace=.57)
    reward(axes[0, 0])
    b.path_panel(axes[0, 1], data, site.split()[0], audit_key)
    b.gantt_panel(axes[1, 0], data)
    for ax, spec in zip([axes[1, 1], axes[2, 0], axes[2, 1], axes[3, 0]], b.STATES):
        b.state_panel(ax, data, spec, audit_key)
    for ax, letter in zip(axes.flat, "abcdefg"):
        b.sublabel(ax, letter)
    axes[3, 1].legend(handles=b.method_handles(methods), loc="upper center", handlelength=4.3,
                      labelspacing=.65, fontsize=14, frameon=True, framealpha=1.0, facecolor="white")
    axes[3, 1].axis("off")
    b.save(fig, f"fig{number:02d}_original_style", out, dpi)
    fig, ax = plt.subplots()
    reward(ax)
    b.save(fig, f"panels/fig{number:02d}_a_reward", out, dpi)
    fig, ax = plt.subplots()
    b.path_panel(ax, data, site.split()[0], audit_key)
    fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.98)
    b.save(fig, f"panels/fig{number:02d}_b_path", out, dpi)
    fig, ax = plt.subplots(figsize=(4, 3))
    b.gantt_panel(ax, data)
    fig.tight_layout()
    b.save(fig, f"panels/fig{number:02d}_c_actions", out, dpi)
    for spec in b.STATES:
        fig, ax = plt.subplots()
        b.state_panel(ax, data, spec, audit_key)
        b.save(fig, f"panels/fig{number:02d}_{spec[0]}_{spec[1].lower()}", out, dpi)
    save_legend(b, out, dpi, f"fig{number:02d}_methods_legend", b.method_handles(methods))


def temporal_frames(b, kind, times):
    times = sorted(set(times))
    filename = "lunarSurfaceTemp.csv" if kind == "temperature" else "lunarSolarLight.csv"
    path = b.source(b.ROOT/"site01"/filename)
    cache = b.ROOT/"outputs/.cache"/f"site01_{kind}_frames.npz"
    cache.parent.mkdir(parents=True, exist_ok=True)
    digest = b.INPUTS[path.relative_to(b.ROOT).as_posix()]["sha256"]
    cached = False
    if cache.exists():
        with np.load(cache) as previous:
            saved_times = previous["times"].tolist()
            if previous["sha256"].item() == digest and set(times).issubset(saved_times):
                frames = previous["frames"][[saved_times.index(t) for t in times]].copy()
                minimum, maximum = float(previous["minimum"]), float(previous["maximum"])
                cached = True
    print(f"Reading {kind} snapshots from original temporal map...", flush=True)
    if not cached:
        frames = np.empty((len(times), 200, 200))
        minimum, maximum = np.inf, -np.inf
        rows = 0
        with path.open() as stream:
            for row, line in enumerate(stream):
                values = np.fromstring(line, sep=",")
                if len(values) != 200*1440 or not np.all(np.isfinite(values)):
                    raise ValueError(f"Unexpected temporal map row: {path}, {row}")
                minimum = min(minimum, float(values.min()))
                maximum = max(maximum, float(values.max()))
                frames[:, row, :] = values.reshape(200, 1440)[:, times].T
                rows += 1
        if rows != 200:
            raise ValueError(f"Expected 200 map rows, found {rows}: {path}")
        # simulation_data_creation.py already converts these CSV values to
        # Celsius before saving; use the stored values without conversion.
        np.savez_compressed(cache, frames=frames, times=times, minimum=minimum,
                            maximum=maximum, sha256=digest)
    FRAME_CACHE[kind] = {t: frame for t, frame in zip(times, frames)}
    b.AUDIT[f"maps/site01/{kind}"] = dict(
        sampled_absolute_times=times, global_min=minimum, global_max=maximum,
        source=path.relative_to(b.ROOT).as_posix(), layout="200 rows x 200 columns x 1440 time steps",
        units="Celsius, original CSV values; simulation_data_creation.py converts before saving" if kind == "temperature" else "original illumination values")
    print(f"Read {kind} snapshots; original range {minimum:.5g} to {maximum:.5g}.", flush=True)
    return minimum, maximum


def endpoint_labels(ax, data, fontsize=9):
    reference = next(iter(data.values()))["path"]
    for label, point in (("Start", reference[0]), ("Goal", reference[-1])):
        xy = (int(point[1])-1, int(point[0])-1)
        offset, va = ((0, -10), "top") if point[0] > 100 else ((0, 10), "bottom")
        ax.scatter(*xy, s=20, facecolors="white", edgecolors="black", linewidths=.8, zorder=6)
        ax.annotate(label, xy, xytext=offset, textcoords="offset points", ha="center", va=va,
                    fontsize=fontsize, bbox=dict(facecolor="white", edgecolor="none", alpha=.9, pad=1), zorder=7)


def snapshot_paths(b, ax, data, step, halo=False, styles=None):
    methods = list(data)
    styles = b.METHOD_PATH_STYLE if styles is None else styles
    interval = b.common_marker_interval([d["path"] for d in data.values()], 10)
    for i, method in enumerate(methods):
        path = data[method]["path"]
        p = path[:step] if len(path) > step else path
        style = styles[method]
        phase, _ = b.marker_every(i, interval, len(methods))
        effects = ([pe.Stroke(linewidth=style["linewidth"]+.75, foreground="white", alpha=.8),
                    pe.Normal()] if halo else None)
        b.path_with_stay_markers(ax, p, style, phase, interval, path_effects=effects)
    endpoint_labels(ax, data)
    ax.set_xlim(-.5, 199.5)
    ax.set_ylim(199.5, -.5)
    ax.axis("off")


def save_legend(b, out, dpi, name, handles, columns=1):
    column_width = 3.9 if columns > 1 else 2.8
    fig, ax = plt.subplots(figsize=(column_width*columns, max(1.2, .45*np.ceil(len(handles)/columns)+.5)))
    ax.legend(handles=handles, loc="center", fontsize=14, handlelength=4.3,
              labelspacing=.65, frameon=True, framealpha=1.0, facecolor="white", ncol=columns)
    ax.axis("off")
    fig.tight_layout()
    b.save(fig, "panels/"+name, out, dpi)


def save_colorbar(b, out, dpi, name, norm, cmap, label):
    fig, ax = plt.subplots(figsize=(1.8, 4.8))
    fig.subplots_adjust(left=.18, right=.38, bottom=.06, top=.96)
    fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=ax, label=label)
    b.save(fig, "panels/"+name, out, dpi)


def figure12(b, out, dpi, temperature_norm, light_norm):
    data = scenario_data(b, "site01", 9)
    styles = {method: dict(style, linewidth=FIG12_PATH_LINEWIDTHS.get(method, FIG12_PATH_LINEWIDTHS["others"]))
              for method, style in b.METHOD_PATH_STYLE.items()}
    times = [100, 200, 300, 400, 500]
    fig, axes = plt.subplots(2, 5, figsize=(18, 7.2))
    fig.subplots_adjust(left=.065, right=.865, bottom=.075, top=.98, wspace=.025, hspace=.025)
    for row, (kind, norm, cmap) in enumerate((("temperature", temperature_norm, "jet"), ("illumination", light_norm, "gray"))):
        for col, t in enumerate(times):
            ax = axes[row, col]
            ax.imshow(FRAME_CACHE[kind][t], cmap=cmap, norm=norm, interpolation="none")
            snapshot_paths(b, ax, data, t, halo=(kind == "temperature"), styles=styles)
            if row == 1:
                ax.text(.5, -.035, f"T={t}", transform=ax.transAxes, ha="center", va="top",
                        fontsize=15, fontfamily="Times New Roman")
    cax = fig.add_axes([.025, .54, .01, .38])
    fig.colorbar(ScalarMappable(norm=temperature_norm, cmap="jet"), cax=cax, label="Surface Temperature (°C)")
    cax.yaxis.set_ticks_position("left")
    cax.yaxis.set_label_position("right")
    cax = fig.add_axes([.89, .095, .01, .37])
    fig.colorbar(ScalarMappable(norm=light_norm, cmap="gray"), cax=cax, label="Shadow and Illumination Intensity")
    fig.legend(handles=b.method_handles(styles=styles), loc="upper left",
               bbox_to_anchor=(.88, .97), fontsize=10, handlelength=3.5, labelspacing=.6)
    b.save(fig, "fig12_original_style", out, dpi)
    for row, (kind, norm, cmap) in enumerate((("temperature", temperature_norm, "jet"), ("illumination", light_norm, "gray"))):
        for col, t in enumerate(times):
            fig, ax = plt.subplots(figsize=(4.8, 4.8))
            fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.93)
            ax.imshow(FRAME_CACHE[kind][t], cmap=cmap, norm=norm, interpolation="none")
            snapshot_paths(b, ax, data, t, halo=(kind == "temperature"), styles=styles)
            ax.set_title(f"T={t}", fontsize=14)
            b.save(fig, f"panels/fig12_{'abcdefghij'[row*5+col]}_{kind}_t{t}", out, dpi)
    save_legend(b, out, dpi, "fig12_methods_legend", b.method_handles(styles=styles))
    save_colorbar(b, out, dpi, "fig12_temperature_colorbar", temperature_norm, "jet", "Surface Temperature (°C)")
    save_colorbar(b, out, dpi, "fig12_illumination_colorbar", light_norm, "gray", "Shadow and Illumination Intensity")
    b.AUDIT["fig12/snapshots"] = dict(
        local_steps=times, start_time=0, original_temperature_gamma=1.1,
        path_slice="path[:step], or full recorded path after completion; same as original code",
        temperature_contrast="original global normalization and gamma, colorbar in Celsius",
        illumination_scale="common linear scale across the five snapshots",
        path_line_widths_pt=FIG12_PATH_LINEWIDTHS,
        trajectory_halo="0.75 pt white under-stroke on temperature maps for legibility")


def figure15(b, out, dpi):
    original = table_time_trajectories(b, 0)
    shifted = table_time_trajectories(b, 660)
    methods = ["Astar_shadow", "PPO", "PPO+GRU"]
    styles, combined = {}, {}
    for method in methods:
        for start_time, data in ((0, original), (660, shifted)):
            name = f"{method}-{start_time}"
            styles[name] = dict(b.METHOD_PATH_STYLE[method], linestyle="-" if start_time == 0 else (0, (6, 2)))
            combined[name] = data[method]
    slope = np.loadtxt(b.source(b.ROOT/"site01/slope.csv"), delimiter=",")
    gray = np.clip(slope/18*255, 0, 255)
    gray[slope >= 18] = 0
    background = np.repeat(gray.astype(np.uint8)[..., None], 3, axis=2)
    handles = [Line2D([], [], **styles[name], label=name, markersize=b.MARKER_SIZE,
                      markerfacecolor="white", markeredgewidth=b.MARKER_EDGE_WIDTH)
               for name in ("PPO+GRU-0", "PPO+GRU-660", "PPO-0", "PPO-660", "Astar_shadow-0", "Astar_shadow-660")]

    def panel(ax, method):
        data = {name: combined[name] for name in (f"{method}-0", f"{method}-660")}
        ax.imshow(background, interpolation="none")
        snapshot_paths(b, ax, data, max(len(d["path"]) for d in data.values()), styles=styles)

    fig, axes = plt.subplots(1, 3, figsize=(15.4, 6.0))
    fig.subplots_adjust(left=.01, right=.99, bottom=.19, top=.95, wspace=.025)
    for ax, method in zip(axes, methods):
        panel(ax, method)
        ax.set_title(method, fontsize=14)
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .015), ncol=3,
               handlelength=4.3, fontsize=12, frameon=True, framealpha=1.0, facecolor="white")
    b.save(fig, "fig15_original_style", out, dpi)
    for letter, method in zip("abc", methods):
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.93)
        panel(ax, method)
        ax.set_title(method, fontsize=14)
        b.save(fig, f"panels/fig15_{letter}_{method.replace('+', '_')}", out, dpi)
    save_legend(b, out, dpi, "fig15_start_time_legend", handles, 2)
    b.AUDIT["fig15/time_styles"] = dict(
        styles=styles, encoding="method colors follow all other figures; time 0 solid, time 660 dense dashed",
        legend="moved below the three maps, exported separately for Visio",
        trajectories="same site01/site01 660 CSV runs as Table 6 and Figs.9/14, using a single run selection")


def figure16(b, out, dpi, norm):
    original = table_time_trajectories(b, 0)
    shifted = table_time_trajectories(b, 660)
    specs = [(0, t, original, 0, (i*4, (i+1)*4)) for i, t in enumerate((300, 400, 500))]
    specs += [(660, t, shifted, 1, (i*3, (i+1)*3)) for i, t in enumerate((200, 1100, 1200, 1300))]

    def panel(ax, start_time, step, data):
        absolute = (start_time+step) % 1440
        ax.imshow(FRAME_CACHE["temperature"][absolute], norm=norm, cmap="jet", interpolation="none")
        selected = {"PPO+GRU": data["PPO+GRU"]}
        styles = {"PPO+GRU": dict(b.METHOD_PATH_STYLE["PPO+GRU"], linewidth=FIG16_PATH_LINEWIDTH)}
        snapshot_paths(b, ax, selected, step, halo=True, styles=styles)
        path = selected["PPO+GRU"]["path"]
        last = path[min(step, len(path))-1]
        ax.scatter(last[1]-1, last[0]-1, s=22, facecolors=b.METHOD_STYLE["PPO+GRU"]["color"],
                   edgecolors="white", linewidths=.7, zorder=5)

    fig = plt.figure(figsize=(15.8, 9.2))
    grid = fig.add_gridspec(2, 12, left=.015, right=.9, top=.93, bottom=.07, hspace=.28, wspace=.08)
    for start_time, step, data, row, (left, right) in specs:
        ax = fig.add_subplot(grid[row, left:right])
        panel(ax, start_time, step, data)
        ax.text(.5, -.04, f"Step={step}", transform=ax.transAxes, ha="center", va="top",
                fontfamily="Times New Roman", fontsize=15)
    fig.text(.455, .965, "Start Time: 0", ha="center", fontsize=17, fontfamily="Times New Roman")
    fig.text(.455, .475, "Start Time: 660", ha="center", fontsize=17, fontfamily="Times New Roman")
    cax = fig.add_axes([.93, .16, .012, .65])
    fig.colorbar(ScalarMappable(norm=norm, cmap="jet"), cax=cax, label="Surface Temperature (°C)")
    b.save(fig, "fig16_original_style", out, dpi)
    for letter, (start_time, step, data, _, _) in zip("abcdefg", specs):
        fig, ax = plt.subplots(figsize=(4.8, 4.8))
        fig.subplots_adjust(left=.02, right=.98, bottom=.02, top=.93)
        panel(ax, start_time, step, data)
        ax.set_title(f"Start Time: {start_time}, Step={step}", fontsize=12)
        b.save(fig, f"panels/fig16_{letter}_start{start_time}_step{step}", out, dpi)
    save_colorbar(b, out, dpi, "fig16_temperature_colorbar", norm, "jet", "Surface Temperature (°C)")
    b.AUDIT["fig16/snapshots"] = dict(
        start_time_0_steps=[300, 400, 500], start_time_660_steps=[200, 1100, 1200, 1300],
        map_time="(local step + start time) modulo 1440", original_temperature_gamma=1.3,
        path_slice="same path[:step] convention as the original code", method=f"PPO+GRU, same blue and {FIG16_PATH_LINEWIDTH:g} pt path width",
        trajectories="same site01/site01 660 PPO+GRU runs as Table 6 and Figs.9/14, using a single run selection",
        trajectory_halo="0.75 pt white under-stroke on colored map, unchanged recorded path")


def render_remaining(b, out: Path, dpi: int, requested):
    for number, (site, offset) in SCENARIOS.items():
        if number in requested:
            comparison(b, out, dpi, number, site, offset)
    if 12 in requested or 16 in requested:
        times = ([100, 200, 300, 400, 500] if 12 in requested else [])
        if 16 in requested:
            times += [300, 400, 500, 860, 320, 420, 520]
        minimum, maximum = temporal_frames(b, "temperature", times)
        if 12 in requested:
            low, high = temporal_frames(b, "illumination", [100, 200, 300, 400, 500])
            figure12(b, out, dpi, PowerNorm(1.1, vmin=minimum, vmax=maximum), Normalize(vmin=low, vmax=high))
    if 15 in requested:
        figure15(b, out, dpi)
    if 16 in requested:
        figure16(b, out, dpi, PowerNorm(1.3, vmin=minimum, vmax=maximum))
