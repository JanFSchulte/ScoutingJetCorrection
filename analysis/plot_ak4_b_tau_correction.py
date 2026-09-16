#!/usr/bin/env python3
"""Combined illustration figure: AK4 response vs pt and vs eta, for b-jets
and gen-matched hadronic-tau-jets, each showing raw / inclusive-corrected /
dedicated-corrected -- the comparison behind the flavour- and tau-dedicated
correction maps (run_flavor_correction.py's per-category closure numbers),
laid out as one 2x2 figure instead of separate per-category PNGs.

Reads results/mc_offline_comparison_hhbbtautau_v7_ak4.npz (has both
off_hadronFlavour, from customiseScoutingNanoWithOfflineJets(), and
off_genVisTau_dr, from build_comparison_table_mc(..., gentau_collection=
"GenVisTau"), so both categories come from the SAME matched-jet population)
and the two per-category response curves already derived by
run_flavor_correction.py (results/flavor_correction_hhbbtautau_v7_ak4/
response_curve_b.npz and results/gentau_correction_hhbbtautau_ak4/
response_curve_tagged.npz), plus recomputes the shared inclusive curve the
same way run_flavor_correction.py does (derive_response_curve on the full
table, default pt/eta bins/min_stat -- must match how those per-category
curves were derived so "inclusive-corrected" here is the exact same
correction they were compared against).

Usage:
    python plot_ak4_b_tau_correction.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import jet_correction, response as response_mod

import argparse

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
    parser.add_argument("--table", default="results/mc_offline_comparison_hhbbtautau_v7_ak4.npz")
    parser.add_argument("--b-curve", default="results/flavor_correction_hhbbtautau_v7_ak4/response_curve_b.npz")
    parser.add_argument("--tau-curve", default="results/gentau_correction_hhbbtautau_ak4/response_curve_tagged.npz")
    parser.add_argument("--output", default="results/ak4_b_tau_correction_summary.png")
    parser.add_argument("--collection-label", default="ScoutingPFJetRecluster2 (plain)",
        help="Scouting collection name shown in the figure title.")
    args = parser.parse_args()

    print("Loading %s" % args.table)
    d = np.load(args.table)
    table = {k: d[k] for k in d.files}
    print("  %d matched jet pairs" % len(table["dr"]))

    inclusive_curve = jet_correction.derive_response_curve(table)
    b_curve = jet_correction.load_response_curve(args.b_curve)
    tau_curve = jet_correction.load_response_curve(args.tau_curve)

    b_mask = table["off_hadronFlavour"] == 5
    tau_mask = table["off_genVisTau_dr"] < 0.4

    b_raw = {k: v[b_mask] for k, v in table.items()}
    tau_raw = {k: v[tau_mask] for k, v in table.items()}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    b_tables = {
        "raw": b_raw,
        "inclusive-corrected": _corrected(b_raw, inclusive_curve),
        "dedicated-corrected": _corrected(b_raw, b_curve),
    }
    _panel(axes[0, 0], axes[0, 1], b_tables, "AK4 b-jets (n=%d)" % b_mask.sum())

    tau_tables = {
        "raw": tau_raw,
        "inclusive-corrected": _corrected(tau_raw, inclusive_curve),
        "dedicated-corrected": _corrected(tau_raw, tau_curve),
    }
    _panel(axes[1, 0], axes[1, 1], tau_tables, "AK4 gen-tau jets (n=%d)" % tau_mask.sum())

    axes[0, 0].legend(fontsize=8, loc="lower right")
    fig.suptitle("HH->bbtautau MC: AK4 %s vs offline response, raw vs inclusive vs dedicated correction"
                 % args.collection_label, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(args.output, dpi=150)
    print("Wrote %s" % args.output)


if __name__ == "__main__":
    main()
