#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Figure 7 Generator
Realtime Latency Decomposition + VR Demo Snapshot

Example 1 (without screenshot):
python figure7_realtime_latency_vr.py \
    --outdir ./output

Example 2 (with screenshot):
python figure7_realtime_latency_vr.py \
    --outdir ./output \
    --screenshot-path ./vr_demo.png
"""

import os
import sys
import traceback
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


# ---------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ---------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------
def build_latency_dataframe(
    server_mean,
    server_std,
    server_min,
    server_max,
    ws_min,
    ws_max,
    unity_min,
    unity_max,
):
    """
    Build latency summary dataframe.
    """

    ws_mean = (ws_min + ws_max) / 2.0
    unity_mean = (unity_min + unity_max) / 2.0

    rows = [
        {
            "component": "Server-side processing",
            "type": "measured",
            "min_ms": server_min,
            "mean_ms": server_mean,
            "max_ms": server_max,
            "std_ms": server_std,
            "note": "Measured latency",
        },
        {
            "component": "WebSocket RTT",
            "type": "estimated",
            "min_ms": ws_min,
            "mean_ms": ws_mean,
            "max_ms": ws_max,
            "std_ms": np.nan,
            "note": "Estimated range",
        },
        {
            "component": "Unity rendering / IK",
            "type": "estimated",
            "min_ms": unity_min,
            "mean_ms": unity_mean,
            "max_ms": unity_max,
            "std_ms": np.nan,
            "note": "Estimated range",
        },
    ]

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Panel A
# ---------------------------------------------------------------------
def plot_latency_panel(
    ax,
    server_mean,
    server_std,
    server_min,
    server_max,
    ws_min,
    ws_max,
    unity_min,
    unity_max,
):
    """
    Left panel:
    latency decomposition
    """

    ws_mean = (ws_min + ws_max) / 2.0
    unity_mean = (unity_min + unity_max) / 2.0

    e2e_min = server_mean + ws_min + unity_min
    e2e_max = server_mean + ws_max + unity_max
    e2e_mean = (e2e_min + e2e_max) / 2.0

    x_server = 0
    x_e2e = 1

    width = 0.55

    # --------------------------------------------------
    # Server-side measured bar
    # --------------------------------------------------
    ax.bar(
        x_server,
        server_mean,
        width=width,
        color="tab:blue",
        edgecolor="black",
        linewidth=1.0,
        label="Measured",
        zorder=3,
    )

    ax.errorbar(
        x_server,
        server_mean,
        yerr=server_std,
        fmt="none",
        ecolor="black",
        capsize=5,
        linewidth=1.2,
        zorder=4,
    )

    ax.text(
        x_server,
        server_mean + server_std + 4,
        f"{server_mean:.2f} ms",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    # --------------------------------------------------
    # E2E estimated stacked bar
    # --------------------------------------------------
    ax.bar(
        x_e2e,
        server_mean,
        width=width,
        color="tab:blue",
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )

    ax.bar(
        x_e2e,
        ws_mean,
        width=width,
        bottom=server_mean,
        color="white",
        hatch="///",
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )

    ax.bar(
        x_e2e,
        unity_mean,
        width=width,
        bottom=server_mean + ws_mean,
        color="white",
        hatch="\\\\\\",
        edgecolor="black",
        linewidth=1.0,
        zorder=3,
    )

    # Range whisker for E2E estimate
    ax.errorbar(
        x_e2e,
        e2e_mean,
        yerr=[
            [e2e_mean - e2e_min],
            [e2e_max - e2e_mean],
        ],
        fmt="none",
        ecolor="black",
        capsize=6,
        linewidth=1.5,
        zorder=5,
    )

    ax.text(
        x_e2e,
        e2e_max + 4,
        f"{int(round(e2e_min))}-{int(round(e2e_max))} ms",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
    )

    # --------------------------------------------------
    # Threshold line
    # --------------------------------------------------
    ax.axhline(
        100,
        linestyle="--",
        linewidth=1.3,
        color="gray",
        zorder=1,
    )

    ax.text(
        1.45,
        101.5,
        "Perceptual threshold (~100 ms)",
        fontsize=9,
        color="dimgray",
        ha="right",
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------
    ax.set_xticks([x_server, x_e2e])
    ax.set_xticklabels(
        [
            "Server-side\nmeasured",
            "End-to-end\nestimated",
        ]
    )

    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3, zorder=0)

    from matplotlib.patches import Patch

    legend_handles = [
        Patch(
            facecolor="tab:blue",
            edgecolor="black",
            label="Measured component",
        ),
        Patch(
            facecolor="white",
            hatch="///",
            edgecolor="black",
            label="Estimated component",
        ),
    ]

    ax.legend(
        handles=legend_handles,
        frameon=False,
        fontsize=9,
        loc="upper left",
    )

    ax.set_title(
        "Latency Decomposition",
        fontsize=11,
        pad=10,
    )

    ax.text(
        -0.18,
        1.04,
        "A",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )


# ---------------------------------------------------------------------
# Panel B
# ---------------------------------------------------------------------
def plot_vr_panel(ax, screenshot_path=None):
    """
    Right panel:
    VR screenshot or placeholder.
    """

    loaded = False

    if screenshot_path is not None and os.path.exists(screenshot_path):
        path_obj = Path(screenshot_path)
        suffix = path_obj.suffix.lower()
        try:
            # If an MP4/MOV file is provided, read the first frame as the still image.
            if suffix in {".mp4", ".mov", ".m4v"}:
                import imageio.v3 as iio

                frame = iio.imread(str(path_obj), index=0)
                ax.imshow(frame)
            else:
                img = plt.imread(str(path_obj))
                ax.imshow(img)
            loaded = True
        except Exception as exc:
            print(f"[WARN] Failed to load visual from {screenshot_path}: {exc}")
            print("[WARN] If you passed a video, install imageio + imageio-ffmpeg in the same Python env.")
            loaded = False

    if not loaded:
        ax.add_patch(
            Rectangle(
                (0, 0),
                1,
                1,
                transform=ax.transAxes,
                facecolor="#efefef",
                edgecolor="black",
                linewidth=1.5,
            )
        )

        ax.text(
            0.5,
            0.55,
            "VR Demo Screenshot\n(placeholder)",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
        )

    callout = (
        "Meta Quest 3\n"
        "WebSocket streaming\n"
        "ONNX inference backend"
    )

    ax.text(
        0.03,
        0.05,
        callout,
        transform=ax.transAxes,
        fontsize=8,
        bbox=dict(
            facecolor="white",
            alpha=0.85,
            edgecolor="black",
        ),
    )

    ax.set_title(
        "VR Demonstration",
        fontsize=11,
        pad=10,
    )

    ax.set_xticks([])
    ax.set_yticks([])

    ax.text(
        -0.08,
        1.04,
        "B",
        transform=ax.transAxes,
        fontsize=14,
        fontweight="bold",
    )

    return loaded


# ---------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------
def save_outputs(fig, df, outdir):
    os.makedirs(outdir, exist_ok=True)

    png_path = os.path.join(
        outdir,
        "figure7_realtime_latency_vr.png",
    )

    pdf_path = os.path.join(
        outdir,
        "figure7_realtime_latency_vr.pdf",
    )

    csv_path = os.path.join(
        outdir,
        "figure7_latency_summary.csv",
    )

    fig.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_path,
        bbox_inches="tight",
    )

    df.to_csv(
        csv_path,
        index=False,
    )

    return png_path, pdf_path, csv_path


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--outdir",
        required=True,
    )

    parser.add_argument(
        "--screenshot-path",
        default=None,
    )

    parser.add_argument(
        "--video-path",
        default=None,
        help="Optional path to VR demo video (.mp4/.mov). If set, this takes priority over --screenshot-path.",
    )

    parser.add_argument(
        "--server-mean",
        type=float,
        default=30.94,
    )

    parser.add_argument(
        "--server-std",
        type=float,
        default=4.67,
    )

    parser.add_argument(
        "--server-min",
        type=float,
        default=18.88,
    )

    parser.add_argument(
        "--server-max",
        type=float,
        default=39.47,
    )

    parser.add_argument(
        "--ws-min",
        type=float,
        default=8,
    )

    parser.add_argument(
        "--ws-max",
        type=float,
        default=12,
    )

    parser.add_argument(
        "--unity-min",
        type=float,
        default=6,
    )

    parser.add_argument(
        "--unity-max",
        type=float,
        default=8,
    )

    parser.add_argument(
        "--n-trials",
        type=int,
        default=200,
    )

    args = parser.parse_args()

    print("\n===== Figure 7 Parameters =====")
    for k, v in vars(args).items():
        print(f"{k}: {v}")

    df = build_latency_dataframe(
        args.server_mean,
        args.server_std,
        args.server_min,
        args.server_max,
        args.ws_min,
        args.ws_max,
        args.unity_min,
        args.unity_max,
    )

    fig = plt.figure(
        figsize=(10, 4.5),
        constrained_layout=True,
    )

    gs = fig.add_gridspec(
        1,
        2,
        width_ratios=[1.15, 1.0],
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    plot_latency_panel(
        ax1,
        args.server_mean,
        args.server_std,
        args.server_min,
        args.server_max,
        args.ws_min,
        args.ws_max,
        args.unity_min,
        args.unity_max,
    )

    visual_path = args.video_path if args.video_path else args.screenshot_path

    screenshot_loaded = plot_vr_panel(
        ax2,
        visual_path,
    )

    fig.suptitle(
        "Figure 7. Real-Time Latency Analysis and VR Demonstration",
        fontsize=13,
        y=1.02,
    )

    png_path, pdf_path, csv_path = save_outputs(
        fig,
        df,
        args.outdir,
    )

    print("\n===== Output Files =====")
    print("PNG :", png_path)
    print("PDF :", pdf_path)
    print("CSV :", csv_path)

    print(
        "\nScreenshot status:",
        "SUCCESS" if screenshot_loaded else "PLACEHOLDER USED",
    )

    plt.close(fig)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
