#!/usr/bin/env python3
"""Illustrate pt-scale and resolution stability of scouting jets (relative to
matched offline jets in the same real collisions) across data-taking eras,
for each of the three baseline scouting jet collections (AK4 plain PUPPI-
reclustered, AK4 CHS, AK8).

Two figures:
  1. results/era_stability_response_resolution.png -- 3 rows (AK4 plain, AK4
     CHS, AK8) x 2 columns (median response = scale, IQR/2 = resolution) vs
     offline jet pt, one line per era, all on shared y-axes per column so
     scale/resolution can be compared directly across collections too.
  2. results/era_stability_trend.png -- scale and resolution at a single
     representative pt window (INCLUSIVE_PT_RANGE) plotted against era in
     chronological order (one point per era per collection) -- the "single
     number vs run period" view standard in JEC validation notes, making any
     drift immediately visible without reading a family of curves.

Inputs: results/data_matching_periods/<era>.npz (AK4 plain, already built by
the original data-matching pipeline for all four eras) and
results/data_matching_periods/<era>_ak4chs.npz / <era>_ak8.npz (CHS/AK8,
built here via run_data_matching_stage3.py against each era's cached
matches.npz -- see that script; avoids repeating the expensive DAS/xrootd
key-scan stages since they don't depend on which jet collection is pulled).

Usage:
    python plot_era_stability.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import response as response_mod

ERAS = ["2024G", "2024H", "2024I", "2025C"]
ERA_COLORS = {"2024G": "#1f77b4", "2024H": "#ff7f0e", "2024I": "#2ca02c", "2025C": "#d62728"}

COLLECTIONS = [
    ("AK4 plain (ScoutingPFJetRecluster2)", "results/data_matching_periods/{era}.npz"),
    ("AK4 CHS (ScoutingPFJetReclusterCHS)", "results/data_matching_periods/{era}_ak4chs.npz"),
    ("AK8 (ScoutingFatPFJetRecluster)", "results/data_matching_periods/{era}_ak8.npz"),
]

# Representative "well-measured plateau" pt window per collection, for the
# trend plot -- AK4 and AK8 don't share one: AK8 (wide-cone, online HLT
# tracking/clustering) has essentially no matched pairs below ~150-175 GeV
# (see the AK4 vs AK8 x-axis ranges in era_stability_response_resolution.png),
# so a single shared window (e.g. AK4's 50-100 GeV) leaves AK8 with zero
# entries and an invisible trend line.
TREND_PT_RANGES = {
    "AK4 plain (ScoutingPFJetRecluster2)": (50.0, 100.0),
    "AK4 CHS (ScoutingPFJetReclusterCHS)": (50.0, 100.0),
    "AK8 (ScoutingFatPFJetRecluster)": (200.0, 300.0),
}

OUT_RESPONSE_RESOLUTION = "results/era_stability_response_resolution.png"
OUT_TREND = "results/era_stability_trend.png"


def _load(path_template, era):
    path = path_template.format(era=era)
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return {k: d[k] for k in d.files}


def make_response_resolution_figure():
    fig, axes = plt.subplots(len(COLLECTIONS), 2, figsize=(12, 12))

    for row, (label, path_template) in enumerate(COLLECTIONS):
        ax_scale, ax_res = axes[row, 0], axes[row, 1]
        for era in ERAS:
            table = _load(path_template, era)
            if table is None:
                continue
            centers, medians, res, counts = response_mod.response_vs_pt_data(table)
            if len(centers) == 0:
                continue
            c = ERA_COLORS[era]
            ax_scale.plot(centers, medians, marker="o", ms=4, label=era, color=c)
            ax_res.plot(centers, res, marker="o", ms=4, label=era, color=c)

        ax_scale.axhline(1.0, color="black", linestyle="--", linewidth=1)
        for ax in (ax_scale, ax_res):
            ax.set_xscale("log")
            ax.set_xlabel("offline jet pt [GeV]")
            ax.grid(alpha=0.3)
        ax_scale.set_ylabel("median response\n(scout pt / offline pt)")
        ax_res.set_ylabel("resolution (IQR/2 of response)")
        ax_scale.set_title("%s: scale vs pt" % label, fontsize=10)
        ax_res.set_title("%s: resolution vs pt" % label, fontsize=10)

    axes[0, 0].legend(fontsize=8, loc="lower right", title="era")
    fig.suptitle("Scouting-vs-offline jet pt scale and resolution stability across data-taking eras",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(OUT_RESPONSE_RESOLUTION, dpi=150)
    print("Wrote %s" % OUT_RESPONSE_RESOLUTION)
    plt.close(fig)


def make_trend_figure():
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5))
    x = np.arange(len(ERAS))
    markers = ["o", "s", "^"]

    ak4_row, ak8_row = axes[0], axes[1]
    ak4_collections = [c for c in COLLECTIONS if not c[0].startswith("AK8")]
    ak8_collections = [c for c in COLLECTIONS if c[0].startswith("AK8")]

    def _fill_row(ax_scale, ax_res, collections):
        for i, (label, path_template) in enumerate(collections):
            pt_lo, pt_hi = TREND_PT_RANGES[label]
            scales, resolutions, valid_x = [], [], []
            for xi, era in zip(x, ERAS):
                table = _load(path_template, era)
                if table is None:
                    continue
                mask = ((table["off_pt"] >= pt_lo) & (table["off_pt"] < pt_hi)
                        & (np.abs(table["scout_eta"]) < 2.5))
                if not np.any(mask):
                    continue
                response = table["scout_pt"][mask] / table["off_pt"][mask]
                q16, q50, q84 = np.percentile(response, [16, 50, 84])
                scales.append(q50)
                resolutions.append(0.5 * (q84 - q16))
                valid_x.append(xi)
            ax_scale.plot(valid_x, scales, marker=markers[i], ms=8, label="%s (%d-%d GeV)" % (label, pt_lo, pt_hi),
                          linewidth=1.5)
            ax_res.plot(valid_x, resolutions, marker=markers[i], ms=8, label=label, linewidth=1.5)
        ax_scale.axhline(1.0, color="black", linestyle="--", linewidth=1)
        for ax in (ax_scale, ax_res):
            ax.set_xticks(x)
            ax.set_xticklabels(ERAS)
            ax.set_xlabel("data-taking era")
            ax.grid(alpha=0.3)
        ax_scale.set_ylabel("median response (scout pt / offline pt)")
        ax_res.set_ylabel("resolution (IQR/2 of response)")
        ax_scale.legend(fontsize=8)

    _fill_row(ak4_row[0], ak4_row[1], ak4_collections)
    _fill_row(ak8_row[0], ak8_row[1], ak8_collections)
    ak4_row[0].set_title("AK4: scale vs era (|eta|<2.5)", fontsize=10)
    ak4_row[1].set_title("AK4: resolution vs era (|eta|<2.5)", fontsize=10)
    ak8_row[0].set_title("AK8: scale vs era (|eta|<2.5)", fontsize=10)
    ak8_row[1].set_title("AK8: resolution vs era (|eta|<2.5)", fontsize=10)

    fig.suptitle("Scouting-vs-offline response/resolution trend across data-taking eras", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUT_TREND, dpi=150)
    print("Wrote %s" % OUT_TREND)
    plt.close(fig)


def main():
    make_response_resolution_figure()
    make_trend_figure()


if __name__ == "__main__":
    main()
