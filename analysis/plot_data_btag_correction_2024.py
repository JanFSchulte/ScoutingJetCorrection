#!/usr/bin/env python3
"""Illustration figure: AK4-collection response vs pt and vs eta for
offline-b-tagged jets (off_btagUParTAK4B >= 0.5, real 2024G+H+I JetMET/Muon
data combined, no gen truth involved), showing raw / inclusive-corrected /
dedicated-corrected -- same layout/colour convention as
plot_ak4_b_tau_correction.py's MC b/tau panels, but for the single
data-only category (no truth flavour to split by) and one row instead of
2x2.

Reads the combined-2024 data comparison table (results/data_matching_periods/
2024_ak4chs_tagger.npz for CHS, 2024_tagger_ak4.npz for plain, 2024_tagger_ak8.npz
for AK8 -- see run_stage3_slurm.py for how these were built) and the
per-category response curve already derived by run_flavor_correction.py
(results/tagger_correction_2024_<collection>_btag/response_curve_tagged.npz).

Usage:
    python plot_data_btag_correction_2024.py \\
        --table results/data_matching_periods/2024_ak4chs_tagger.npz \\
        --tagged-curve results/tagger_correction_2024_ak4chs_btag/response_curve_tagged.npz \\
        --score-field off_btagUParTAK4B --collection-label "AK4 CHS" \\
        --output results/data2024_ak4chs_btag_correction_summary.png
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import jet_correction, response as response_mod

ETA_PT_RANGE = (30.0, None)


def _corrected(table, curve):
    out = dict(table)
    # setdefault, not assign: keeps the raw scout_pt fixed for
    # response_vs_pt_data/vs_eta_data's min_scout_pt selection, so raw and
    # corrected curves are computed over the SAME set of jets -- see
    # response.py's response_vs_pt_data docstring.
    out.setdefault("scout_pt_raw", table["scout_pt"])
    out["scout_pt"] = jet_correction.apply_correction_inverted(table["scout_pt"], table["scout_eta"], curve)
    return out


def _panel(ax_pt, ax_eta, subtables, title):
    colors = {"raw": "#888888", "inclusive-corrected": "#1f77b4", "dedicated-corrected": "#d62728"}
    for name, sub in subtables.items():
        c = colors[name]
        centers, medians, _, _ = response_mod.response_vs_pt_data(sub)
        ax_pt.plot(centers, medians, marker="o", ms=4, label=name, color=c)

        centers_e, medians_e, _, _ = response_mod.response_vs_eta_data(sub, pt_range=ETA_PT_RANGE)
        ax_eta.plot(centers_e, medians_e, marker="o", ms=4, label=name, color=c)

    for ax in (ax_pt, ax_eta):
        ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
        ax.grid(alpha=0.3)
    ax_pt.set_xlabel("offline jet pt [GeV]")
    ax_pt.set_ylabel("median response (scout/offline pt)")
    ax_pt.set_xscale("log")
    ax_pt.set_title("%s: response vs pt" % title)
    ax_eta.set_xlabel("offline jet eta")
    ax_eta.set_title("%s: response vs eta (offline pt > %.0f GeV)" % (title, ETA_PT_RANGE[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True)
    parser.add_argument("--tagged-curve", required=True)
    parser.add_argument("--score-field", default="off_btagUParTAK4B")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--collection-label", default="AK4")
    parser.add_argument("--category-label", default="b-tag")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Loading %s" % args.table)
    d = np.load(args.table)
    table = {k: d[k] for k in d.files}
    print("  %d matched jet pairs" % len(table["dr"]))

    inclusive_curve = jet_correction.derive_response_curve(table)
    tagged_curve = jet_correction.load_response_curve(args.tagged_curve)

    mask = table[args.score_field] >= args.score_threshold
    raw = {k: v[mask] for k, v in table.items()}

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))

    tagged_tables = {
        "raw": raw,
        "inclusive-corrected": _corrected(raw, inclusive_curve),
        "dedicated-corrected": _corrected(raw, tagged_curve),
    }
    _panel(axes[0], axes[1], tagged_tables,
           "%s %s jets (n=%d)" % (args.collection_label, args.category_label, int(mask.sum())))

    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle("Real 2024G+H+I data: %s vs offline response, raw vs inclusive vs dedicated correction"
                 % args.collection_label, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(args.output, dpi=150)
    print("Wrote %s" % args.output)


if __name__ == "__main__":
    main()
