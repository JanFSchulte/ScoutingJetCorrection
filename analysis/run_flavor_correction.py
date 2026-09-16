#!/usr/bin/env python3
"""Derive a dedicated per-flavour response correction (jet_correction.
derive_response_curve_by_category) and compare it against applying the
INCLUSIVE correction to each flavour -- the check that motivated this: an
inclusive AK4 correction (fit mostly to light/gluon jets, the majority
species in most samples) left a real ~4% residual pt bias specifically on
b-jets in the HH->bbtautau MC sample, worse than light jets' ~1% residual
under the same inclusive map.

Two ways to split into categories:
  1. Truth flavour (MC only, default): a comparison table with an integer
     flavour field already attached (e.g. off_hadronFlavour, added to
     OfflineJet/OfflineFatJet by scoutingToMiniAODDerivedCollections_cff.py
     and carried through by run_mc_offline_comparison.py's --off-fields
     hadronFlavour). Categories = b (==5) vs light (==0).
  2. Tagger score threshold (--score-field/--score-threshold): works on data
     too, since it's a reconstructed discriminator, not truth -- e.g.
     off_btagUParTAK4B (offline UParT b-tag score) or
     off_globalParT3_Xtauhtauh (offline AK8 tau-vs-QCD score), both already
     present in central JetMET/Muon NanoAOD and readable via
     data_matching.build_comparison_table's off_fields/scout_fields (no new
     production needed). Categories = tagged (score >= threshold) vs
     untagged (score < threshold).

Usage:
    # MC, truth flavour:
    python run_flavor_correction.py \\
        --table results/mc_offline_comparison_hhbbtautau_ak4.npz --label HHbbtautau_AK4 \\
        --outdir results/flavor_correction_hhbbtautau_ak4

    # Data, tagger score:
    python run_flavor_correction.py \\
        --table results/data_matching_periods/2025C_tagger_ak4.npz --label Data2025C_AK4 \\
        --outdir results/tagger_correction_2025C_ak4 \\
        --score-field off_btagUParTAK4B --score-threshold 0.5
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

CATEGORIES = {
    "b": lambda f: f == 5,
    "light": lambda f: f == 0,
}


def _corrected_table(table, curve):
    out = dict(table)
    # setdefault, not assign: preserves the TRUE original raw scout_pt even
    # if `table` is itself already a corrected table (so response_vs_pt_data/
    # vs_eta_data's min_scout_pt selection stays fixed to the real raw
    # population, not whatever scout_pt happens to hold after correction --
    # see response.py's response_vs_pt_data docstring for why this matters).
    out.setdefault("scout_pt_raw", table["scout_pt"])
    out["scout_pt"] = jet_correction.apply_correction_inverted(table["scout_pt"], table["scout_eta"], curve)
    return out


def _closure(table, pt_bins, eta_pt_range):
    _, m_pt, _, _ = response_mod.response_vs_pt_data(table, pt_bins=pt_bins)
    _, m_eta, _, _ = response_mod.response_vs_eta_data(table, pt_range=eta_pt_range)
    max_pt = np.max(np.abs(m_pt - 1.0)) if len(m_pt) else float("nan")
    max_eta = np.max(np.abs(m_eta - 1.0)) if len(m_eta) else float("nan")
    return max_pt, max_eta


def _plot_category_vs_pt(tables, outdir, label, category, pt_bins):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, table in tables.items():
        centers, medians, res, counts = response_mod.response_vs_pt_data(table, pt_bins=pt_bins)
        if len(centers) == 0:
            continue
        axes[0].plot(centers, medians, marker="o", label=name)
        axes[1].plot(centers, res, marker="o", label=name)
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("offline jet pt [GeV]")
    axes[0].set_ylabel("median response (scout pt / offline pt)")
    axes[0].set_title("%s (%s) response vs pt" % (label, category))
    axes[1].set_xlabel("offline jet pt [GeV]")
    axes[1].set_ylabel("resolution (IQR/2 of response)")
    axes[1].set_title("%s (%s) resolution vs pt" % (label, category))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "response_resolution_vs_pt_%s.png" % category), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True,
        help=".npz comparison table with an off_hadronFlavour field (e.g. from "
             "run_mc_offline_comparison.py --off-fields hadronFlavour partonFlavour)")
    parser.add_argument("--label", default="jets")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--category-field", default="off_hadronFlavour",
        help="Truth-flavour mode (default): integer field, categories are b (==5) vs "
             "light (==0). Ignored if --score-field is given.")
    parser.add_argument("--score-field", default=None,
        help="Tagger-score mode instead of truth flavour: a continuous discriminator "
             "field (e.g. off_btagUParTAK4B, off_globalParT3_Xtauhtauh). Categories are "
             "tagged (score >= --score-threshold) vs untagged. Works on data.")
    parser.add_argument("--score-threshold", type=float, default=0.5,
        help="Threshold for --score-field (default 0.5).")
    parser.add_argument("--score-op", choices=["ge", "le"], default="ge",
        help="Comparison used to define 'tagged' for --score-field: 'ge' "
             "(score >= threshold, default -- tagger discriminators, higher "
             "is more signal-like) or 'le' (score <= threshold -- e.g. a dR-"
             "to-gen-object field, where SMALLER means a better match).")
    parser.add_argument("--pt-bins", type=float, nargs="+", default=None)
    parser.add_argument("--eta-bins", type=float, nargs="+", default=None)
    parser.add_argument("--eta-pt-range", type=float, nargs=2, default=[30.0, None],
                         metavar=("PT_LO", "PT_HI"))
    parser.add_argument("--min-stat", type=int, default=50)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("Loading %s" % args.table)
    d = np.load(args.table)
    table = {k: d[k] for k in d.files}

    if args.score_field:
        category_field = args.score_field
        thr = args.score_threshold
        if args.score_op == "ge":
            categories = {"tagged": lambda s, t=thr: s >= t, "untagged": lambda s, t=thr: s < t}
        else:
            categories = {"tagged": lambda s, t=thr: s <= t, "untagged": lambda s, t=thr: s > t}
    else:
        category_field = args.category_field
        categories = CATEGORIES

    if category_field not in table:
        raise SystemExit("%s has no field %r -- rebuild the comparison table with the right "
                          "--off-fields/--scout-fields (see run_mc_offline_comparison.py / "
                          "data_matching.build_comparison_table)." % (args.table, category_field))
    print("  %d matched jet pairs" % len(table["dr"]))

    pt_bins = np.array(args.pt_bins) if args.pt_bins else None
    eta_bins = np.array(args.eta_bins) if args.eta_bins else None
    eta_pt_range = tuple(args.eta_pt_range)

    inclusive_curve = jet_correction.derive_response_curve(
        table, pt_bins=pt_bins, eta_bins=eta_bins, min_stat=args.min_stat)

    field = table[category_field]
    per_category_curves = jet_correction.derive_response_curve_by_category(
        table, categories, category_field=category_field,
        pt_bins=pt_bins, eta_bins=eta_bins, min_stat=args.min_stat)

    print("%s closure (max |median response - 1|):" % args.label)
    print("  %-28s %10s %10s" % ("", "vs pt", "vs eta"))
    for name, predicate in categories.items():
        mask = predicate(field)
        sub_raw = {k: v[mask] for k, v in table.items()}
        n = int(np.count_nonzero(mask))

        max_pt_raw, max_eta_raw = _closure(sub_raw, pt_bins, eta_pt_range)
        print("  %-16s raw (n=%-9d) %9.4f %9.4f" % (name, n, max_pt_raw, max_eta_raw))

        sub_inclusive = _corrected_table(sub_raw, inclusive_curve)
        max_pt_inc, max_eta_inc = _closure(sub_inclusive, pt_bins, eta_pt_range)
        print("  %-16s inclusive-corr     %9.4f %9.4f" % (name, max_pt_inc, max_eta_inc))

        sub_dedicated = _corrected_table(sub_raw, per_category_curves[name])
        max_pt_ded, max_eta_ded = _closure(sub_dedicated, pt_bins, eta_pt_range)
        print("  %-16s dedicated-corr      %9.4f %9.4f" % (name, max_pt_ded, max_eta_ded))

        jet_correction.save_response_curve(
            os.path.join(args.outdir, "response_curve_%s.npz" % name), per_category_curves[name])

        _plot_category_vs_pt(
            {"raw": sub_raw, "inclusive-corrected": sub_inclusive, "dedicated-corrected": sub_dedicated},
            args.outdir, args.label, name, pt_bins)

    print("Wrote per-category response curves and plots to %s" % args.outdir)


if __name__ == "__main__":
    main()
