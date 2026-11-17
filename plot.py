#!/usr/bin/env python3
"""Generate separate fixed-cache and standard-test throughput figures.

Install requirements-plot.txt, then run: python plot.py --preview
Use --language zh for Chinese labels and --font to select a Chinese font file.
"""

import argparse
import hashlib
from io import BytesIO
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FixedLocator, ScalarFormatter
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "results/2026-11-a100-sharing"
DEFAULT_OUTPUT = ROOT / "out"
INPUT_FILES = ("controlled.csv", "fullstack.csv", "fullstack-unlimited.csv")
# Colorblind-friendly colors, with markers and line styles for monochrome copies.
SERIES = [
    ("Native single", "原生单实例", "#999999", "o", "-"),
    ("Native pair", "原生双实例", "#333333", "s", "-"),
    ("HAMi pair (50%)", "HAMi 50%", "#0072B2", "^", "--"),
    ("HAMi pair", "HAMi 不限算力", "#009E73", "x", ":"),
    ("KAI + GPU Fractioning pair", "GPU Fractioning（MPS v3）", "#D55E00", "D", "-."),
]


def configure_fonts(font_path, language):
    chinese = None
    if font_path:
        font_manager.fontManager.addfont(font_path)
        chinese = font_manager.FontProperties(fname=font_path).get_name()
    elif language == "zh":
        available = {f.name for f in font_manager.fontManager.ttflist}
        chinese = next(
            (name for name in ["Noto Serif CJK SC", "Source Han Serif SC", "Songti SC", "STSong", "PingFang SC"] if name in available),
            None,
        )
        if chinese is None:
            raise ValueError("Install a Chinese font or supply --font /path/to/font.otf")
    plt.rcParams.update({
        "font.family": [name for name in ["DejaVu Serif", chinese] if name],
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "axes.linewidth": 0.8,
        "axes.unicode_minus": False,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 11,
        "legend.frameon": False,
        "lines.linewidth": 1.4,
        "svg.fonttype": "path",
        "svg.hashsalt": "hami-gpu-sharing-benchmarks",
        "savefig.facecolor": "white",
    })


def save_figure(figure, output_dir, name, title, description, plotted, preview):
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"gpu-sharing-benchmarks-{name}.svg"
    figure.savefig(output, format="svg", metadata={
        "Title": title,
        "Description": description + "\n" + json.dumps(plotted, ensure_ascii=False, sort_keys=True),
        "Date": None,
    })
    # Keep generated SVGs clean in Git while preserving deterministic output.
    output.write_text("\n".join(line.rstrip() for line in output.read_text(encoding="utf-8").splitlines()) + "\n", encoding="utf-8")
    if preview:
        figure.savefig(output.with_suffix(".png"), dpi=180)
    plt.close(figure)
    print(f"Saved {output}")


def expand_y_limits(axis, minimum_top):
    bottom, top = axis.get_ylim()
    bottom, top = min(0, bottom), max(minimum_top, top)
    axis.set_ylim(bottom, top)
    if axis.texts:
        axis.figure.canvas.draw()
        renderer = axis.figure.canvas.get_renderer()
        for label in axis.texts:
            # Text has a fixed display height; reserve that fraction of the
            # axes above its data anchor even when the data range expands.
            height_fraction = label.get_window_extent(renderer).height / axis.bbox.height
            required_top = bottom + (label.get_position()[1] - bottom) / (1 - height_fraction)
            top = max(top, required_top)
        axis.set_ylim(bottom, top)


def draw(data_dir, output_dir, preview, font_path, language):
    configure_fonts(font_path, language)
    labels = {
        "en": {
            "fixed": "Fixed cache", "regular": "Standard test",
            "throughput": "Output throughput (tokens/s/GPU)",
            "concurrency": "Total concurrency per GPU (log scale)",
            "bars": ["Native\nsingle", "Native\npair", "HAMi\n50%", "MPS\nv2"],
            "series": ["Native single", "Native pair", "HAMi 50%", "HAMi unlimited", "GPU Fractioning (MPS v3)"],
        },
        "zh": {
            "fixed": "固定缓存", "regular": "常规测试",
            "throughput": "输出吞吐 (tokens/s/GPU)",
            "concurrency": "每卡总并发（对数刻度）",
            "bars": ["原生\n单实例", "原生\n双实例", "HAMi\n50%", "MPS\nv2"],
            "series": [item[1] for item in SERIES],
        },
    }[language]
    sources = {name: (data_dir / name).read_bytes() for name in INPUT_FILES}
    controlled = pd.read_csv(BytesIO(sources["controlled.csv"])).set_index(
        ["mode", "configuration"]
    )
    regular = pd.read_csv(BytesIO(sources["fullstack.csv"]))
    unlimited = pd.read_csv(BytesIO(sources["fullstack-unlimited.csv"]))
    regular = pd.concat([
        regular,
        unlimited[unlimited["configuration"] == "HAMi pair"],
    ]).set_index(
        ["mode", "configuration", "total_concurrency_per_gpu"],
    )
    controlled = controlled.sort_index()
    regular = regular.sort_index()
    if not controlled.index.is_unique or not regular.index.is_unique:
        raise ValueError("Input CSVs contain duplicate mode/configuration/concurrency rows")
    # Validate the complete plotted dataset before creating or replacing artifacts.
    for mode in ["Graph", "Eager"]:
        for key in ["Native single", "Native pair", "HAMi pair (50%)", "Legacy MPS pair"]:
            row = controlled.loc[(mode, key)]
            for column in ["tps_per_gpu_mean", "tps_per_gpu_sample_std"]:
                value = float(row[column])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {column}: {mode}, {key}")
        for key, *_ in SERIES:
            for concurrency in [2, 4, 8, 16, 32]:
                value = float(regular.loc[(mode, key, concurrency), "tps_per_gpu"])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid throughput: {mode}, {key}, {concurrency}")
    fixed_figure, fixed_axes = plt.subplots(1, 2, figsize=(10.5, 4.1))
    fixed_figure.subplots_adjust(left=0.09, right=0.98, top=0.86, bottom=0.18, wspace=0.27)
    regular_figure, regular_axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    regular_figure.subplots_adjust(left=0.09, right=0.98, top=0.88, bottom=0.32, wspace=0.27)
    hashes = {name: hashlib.sha256(sources[name]).hexdigest() for name in INPUT_FILES}
    fixed_plotted = {
        "source_sha256": {"controlled.csv": hashes["controlled.csv"]},
        "controlled": [],
    }
    regular_plotted = {
        "source_sha256": {name: hashes[name] for name in INPUT_FILES[1:]},
        "regular": [],
    }
    fixed_keys = ["Native single", "Native pair", "HAMi pair (50%)", "Legacy MPS pair"]
    bar_colors = ["#BBBBBB", "#555555", "#0072B2", "#D55E00"]

    for column, mode in enumerate(["Graph", "Eager"]):
        axis = fixed_axes[column]
        values = controlled.loc[mode].reindex(fixed_keys)
        bars = axis.bar(
            range(4), values["tps_per_gpu_mean"],
            yerr=values["tps_per_gpu_sample_std"], width=0.62,
            color=bar_colors, edgecolor="black", linewidth=0.6,
            capsize=3, error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
        bars[0].set_hatch("///")
        bars[2].set_hatch("..")
        axis.set_xticks(range(4), labels["bars"])
        axis.set_title(f"({'ab'[column]}) {labels['fixed']} · {mode}", pad=12)
        axis.set_ylabel(labels["throughput"])
        for x, (key, row) in enumerate(values.iterrows()):
            mean, std = float(row["tps_per_gpu_mean"]), float(row["tps_per_gpu_sample_std"])
            axis.text(x, mean + std + 13, f"{mean:.2f}", ha="center", va="bottom", fontsize=10)
            bars[x].set_gid(f"controlled-{mode}-{x}")
            fixed_plotted["controlled"].append({"mode": mode, "configuration": key, "mean": mean, "sample_std": std})
        expand_y_limits(axis, 700)
        if axis.get_ylim() == (0, 700):
            axis.set_yticks(range(0, 701, 100))
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="0.88", linewidth=0.5)

    levels = [2, 4, 8, 16, 32]
    handles = []
    for column, mode in enumerate(["Graph", "Eager"]):
        axis = regular_axes[column]
        for series_index, (key, label, color, marker, style) in enumerate(SERIES):
            values = regular.loc[(mode, key)].reindex(levels)["tps_per_gpu"]
            line, = axis.plot(
                levels, values, label=labels["series"][series_index], color=color, marker=marker,
                linestyle=style, markersize=5, markerfacecolor="white",
                markeredgewidth=1, clip_on=False,
            )
            line.set_gid(f"regular-{mode}-{series_index}")
            if column == 0:
                handles.append(line)
            regular_plotted["regular"].extend({"mode": mode, "configuration": key, "concurrency": int(c), "tps": float(v)} for c, v in values.items())
        axis.set_xscale("log", base=2)
        axis.set_xlim(1.8, 35.5)
        axis.xaxis.set_major_locator(FixedLocator(levels))
        axis.xaxis.set_major_formatter(ScalarFormatter())
        axis.minorticks_off()
        expand_y_limits(axis, 1200 if mode == "Graph" else 700)
        axis.set_title(f"({'ab'[column]}) {labels['regular']} · {mode}", pad=12)
        axis.set_xlabel(labels["concurrency"])
        axis.set_ylabel(labels["throughput"])
        axis.set_axisbelow(True)
        axis.grid(axis="y", color="0.88", linewidth=0.5)

    regular_figure.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.52, 0.025), ncol=3, columnspacing=1.5, handlelength=2.5)
    save_figure(
        fixed_figure, output_dir, "fixed-cache", "GPU sharing throughput: fixed-cache test",
        "Fixed 6 GiB KV cache per instance, concurrency 8 per GPU; "
        "bars show the three-trial mean and error bars show +/-1 sample standard deviation.",
        fixed_plotted, preview,
    )
    save_figure(
        regular_figure, output_dir, "standard", "GPU sharing throughput: standard test",
        "Automatic KV cache, concurrency 2/4/8/16/32 per GPU, one trial per configuration and concurrency level.",
        regular_plotted, preview,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", action="store_true", help="Also save a PNG preview of each figure")
    parser.add_argument("--font", help="Optional path to a Chinese font file")
    parser.add_argument("--language", choices=["en", "zh"], default="en")
    args = parser.parse_args()
    try:
        draw(args.data_dir, args.output_dir, args.preview, args.font, args.language)
    except (ValueError, KeyError, OSError) as exc:
        parser.exit(1, f"plot.py: {exc}\n")
