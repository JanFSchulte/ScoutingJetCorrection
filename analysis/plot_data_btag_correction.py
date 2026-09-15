#!/usr/bin/env python3
"""Combined illustration figure: AK4 DATA response vs pt and vs eta for
offline-b-tagged jets (off_btagUParTAK4B >= 0.5, real 2025C JetMET/Muon
data, no gen truth involved), showing raw / inclusive-corrected / dedicated
-corrected -- same layout as plot_ak4_b_tau_correction.py's MC figure, one
row per scouting jet collection (plain PUPPI-reclustered vs CHS).

Reads the two data comparison tables built by run_data_matching_stage3.py
(results/data_matching_periods/2025C_tagger_ak4.npz for plain,
2025C_ak4chs_tagger.npz for CHS -- both carry off_btagUParTAK4B, the offline
central-NanoAOD b-tag discriminator, so the b-tag definition is identical
across rows) and the per-category response curves already derived by
run_flavor_correction.py for each (results/tagger_correction_2025C_ak4_btag/
response_curve_tagged.npz and .../2025C_ak4chs_btag/response_curve_tagged.npz).

Usage:
    python plot_data_btag_correction.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import jet_correction, response as response_mod

ETA_PT_RANGE = (30.0, None)
SCORE_THRESHOLD = 0.5

ROWS = [
    ("plain (ScoutingPFJetRecluster2)",
     "results/data_matching_periods/2025C_tagger_ak4.npz",
     "results/tagger_correction_2025C_ak4_btag/response_curve_tagged.npz"),
    ("CHS (ScoutingPFJetReclusterCHS)",
     "results/data_matching_periods/2025C_ak4chs_tagger.npz",
     "results/tagger_correction_2025C_ak4chs_btag/response_curve_tagged.npz"),
]

OUTPUT = "results/data_2025C_ak4_btag_plain_vs_chs.png"


def _corrected(table, curve):
    out = dict(table)
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
    ax_pt.set_title("%s\nresponse vs pt" % title, fontsize=10)
    ax_eta.set_xlabel("offline jet eta")
    ax_eta.set_title("response vs eta (offline pt > %.0f GeV)" % ETA_PT_RANGE[0], fontsize=10)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    for row, (label, table_path, curve_path) in enumerate(ROWS):
        print("Loading %s" % table_path)
        d = np.load(table_path)
        table = {k: d[k] for k in d.files}
        print("  %d matched jet pairs" % len(table["dr"]))

        inclusive_curve = jet_correction.derive_response_curve(table)
        dedicated_curve = jet_correction.load_response_curve(curve_path)

        mask = table["off_btagUParTAK4B"] >= SCORE_THRESHOLD
        raw = {k: v[mask] for k, v in table.items()}

        tables = {
            "raw": raw,
            "inclusive-corrected": _corrected(raw, inclusive_curve),
            "dedicated-corrected": _corrected(raw, dedicated_curve),
        }
        _panel(axes[row, 0], axes[row, 1], tables, "%s, b-tagged data (n=%d)" % (label, mask.sum()))

    axes[0, 0].legend(fontsize=8, loc="upper right")
    fig.suptitle("2025C data: AK4 offline-b-tagged jets, raw vs inclusive vs dedicated correction",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(OUTPUT, dpi=150)
    print("Wrote %s" % OUTPUT)


if __name__ == "__main__":
    main()
