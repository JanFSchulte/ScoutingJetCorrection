#!/usr/bin/env python3
"""Response/resolution-vs-pt and vs-eta plots for a scout-vs-offline matched
comparison table (from run_mc_offline_comparison.py or run_data_matching.py),
plus a pt/eta-binned scale-factor correction (jet_correction.py) derived on
the same table, with the corrected-jet curve overlaid for comparison.

Usage:
    python run_offline_comparison_plots.py \\
        --table mc_offline_comparison_ak4.npz --label AK4 --outdir plots_ak4/

    python run_offline_comparison_plots.py \\
        --table mc_offline_comparison_ak8.npz --label AK8 --outdir plots_ak8/ \\
        --pt-bins 200 250 300 400 500 700 1000
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


def _corrected_table(table, corr):
    """A copy of table with scout_pt replaced by the SF-corrected pt, so the
    existing response_vs_pt_data/response_vs_eta_data can be reused unchanged."""
    out = dict(table)
    out["scout_pt"] = jet_correction.apply_correction(table["scout_pt"], table["scout_eta"], corr)
    return out


def _fit_corrected_table(table, fit_corr):
    out = dict(table)
    out["scout_pt"] = jet_correction.apply_correction_fit(table["scout_pt"], table["scout_eta"], fit_corr)
    return out


def _iterative_corrected_table(table, corrections):
    out = dict(table)
    out["scout_pt"] = jet_correction.apply_correction_iterative(table["scout_pt"], table["scout_eta"], corrections)
    return out


def _inverted_corrected_table(table, curve):
    out = dict(table)
    out["scout_pt"] = jet_correction.apply_correction_inverted(table["scout_pt"], table["scout_eta"], curve)
    return out


def _plot_vs_pt(tables, outdir, label, pt_bins):
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
    axes[0].set_title("%s response vs pt" % label)
    axes[1].set_xlabel("offline jet pt [GeV]")
    axes[1].set_ylabel("resolution (IQR/2 of response)")
    axes[1].set_title("%s resolution vs pt" % label)
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "response_resolution_vs_pt.png"), dpi=150)
    plt.close(fig)


def _plot_vs_eta(tables, outdir, label, eta_bins, pt_range):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, table in tables.items():
        centers, medians, res, counts = response_mod.response_vs_eta_data(
            table, eta_bins=eta_bins, pt_range=pt_range)
        if len(centers) == 0:
            continue
        axes[0].plot(centers, medians, marker="o", label=name)
        axes[1].plot(centers, res, marker="o", label=name)
    pt_lo, pt_hi = pt_range
    pt_label = "%s<pt<%s GeV" % (pt_lo if pt_lo is not None else "-inf",
                                  pt_hi if pt_hi is not None else "inf")
    axes[0].axhline(1.0, color="gray", linestyle="--", linewidth=1)
    axes[0].set_xlabel("scout jet eta")
    axes[0].set_ylabel("median response (scout pt / offline pt)")
    axes[0].set_title("%s response vs eta  (%s)" % (label, pt_label))
    axes[1].set_xlabel("scout jet eta")
    axes[1].set_ylabel("resolution (IQR/2 of response)")
    axes[1].set_title("%s resolution vs eta  (%s)" % (label, pt_label))
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "response_resolution_vs_eta.png"), dpi=150)
    plt.close(fig)


def _plot_correction_maps(corr, outdir, label):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, grid, title in ((axes[0], corr["sf"], "scale factor SF(pt,eta)"),
                             (axes[1], corr["resolution"], "residual resolution (IQR/2, diagnostic only)")):
        im = ax.pcolormesh(corr["pt_bins"], corr["eta_bins"], grid, shading="flat")
        ax.set_xlabel("scout jet pt [GeV]")
        ax.set_ylabel("scout jet eta")
        ax.set_title("%s %s" % (label, title))
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "correction_maps.png"), dpi=150)
    plt.close(fig)


def _plot_fit_qa(table, fit_corr, outdir, label):
    """QA plot: binned median response (the fit input data) vs the fitted
    curve, one panel per eta bin, so a bad fit is visible directly rather
    than only showing up as a closure regression downstream."""
    eta_bins = fit_corr["eta_bins"]
    n_eta = len(eta_bins) - 1
    ncols = min(5, n_eta)
    nrows = -(-n_eta // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 2.8 * nrows), squeeze=False)
    pt_lo, pt_hi = fit_corr["pt_fit_range"]
    pt_fit_bins = np.geomspace(pt_lo, pt_hi, 41)
    ieta = np.digitize(table["scout_eta"], eta_bins) - 1
    response = table["scout_pt"] / table["off_pt"]
    pt_plot = np.geomspace(pt_lo, pt_hi, 200)
    for ie in range(n_eta):
        ax = axes[ie // ncols][ie % ncols]
        sel = ieta == ie
        centers, medians, counts = jet_correction._median_vs_pt(
            table["scout_pt"][sel], response[sel], pt_fit_bins, 20)
        ax.plot(centers, medians, "o", markersize=3, label="binned median")
        if fit_corr["fit_ok"][ie]:
            p = fit_corr["params"][ie]
            ax.plot(pt_plot, jet_correction._l2rel_func(pt_plot, *p), "-", label="fit")
        ax.set_xscale("log")
        ax.set_title("eta [%.2f, %.2f)" % (eta_bins[ie], eta_bins[ie + 1]), fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes[0][0].legend(fontsize=6)
    fig.suptitle("%s fit QA: response(pt) per eta bin" % label)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fit_qa.png"), dpi=150)
    plt.close(fig)


def _print_iterative_convergence(table, corrections, pt_bins, eta_bins, eta_pt_range, label):
    """Closure after each round of derive_correction_iterative, so the
    convergence (or lack of it) is visible round by round, not just at the
    final round."""
    print("%s iterative closure by round (max |median response - 1|):" % label)
    for k in range(1, len(corrections) + 1):
        partial = _iterative_corrected_table(table, corrections[:k])
        _, medians_pt, _, _ = response_mod.response_vs_pt_data(partial, pt_bins=pt_bins)
        _, medians_eta, _, _ = response_mod.response_vs_eta_data(
            partial, eta_bins=eta_bins, pt_range=eta_pt_range)
        max_pt = np.max(np.abs(medians_pt - 1.0)) if len(medians_pt) else float("nan")
        max_eta = np.max(np.abs(medians_eta - 1.0)) if len(medians_eta) else float("nan")
        print("  round %d  vs pt: %.4f   vs eta: %.4f" % (k, max_pt, max_eta))


def _print_closure(tables, pt_bins, eta_bins, eta_pt_range, label):
    """Max |median response - 1| over the vs-pt and vs-eta binnings, raw vs
    SF-corrected -- a quick numeric handle on how much bias the SF removes
    and how much is left over, independent of eyeballing the plots."""
    print("%s closure (max |median response - 1|):" % label)
    for name, table in tables.items():
        _, medians_pt, _, _ = response_mod.response_vs_pt_data(table, pt_bins=pt_bins)
        _, medians_eta, _, _ = response_mod.response_vs_eta_data(
            table, eta_bins=eta_bins, pt_range=eta_pt_range)
        max_pt = np.max(np.abs(medians_pt - 1.0)) if len(medians_pt) else float("nan")
        max_eta = np.max(np.abs(medians_eta - 1.0)) if len(medians_eta) else float("nan")
        print("  %-14s vs pt: %.4f   vs eta: %.4f" % (name, max_pt, max_eta))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--table", required=True, help=".npz comparison table from "
                         "run_mc_offline_comparison.py / run_data_matching.py")
    parser.add_argument("--label", default="jets", help="Label used in plot titles/filenames")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--pt-bins", type=float, nargs="+", default=None,
                         help="Bin edges for the vs-pt plots and the correction map's pt axis")
    parser.add_argument("--eta-bins", type=float, nargs="+", default=None,
                         help="Bin edges for the vs-eta plots and the correction map's eta axis")
    parser.add_argument("--eta-pt-range", type=float, nargs=2, default=[30.0, None],
                         metavar=("PT_LO", "PT_HI"), help="offline pt window used for the "
                         "vs-eta plots (default: pt>30 GeV)")
    parser.add_argument("--min-stat", type=int, default=50,
                         help="Minimum matched pairs per (pt,eta) bin to derive a "
                              "correction there (default 50); below this, SF=1 (no-op)")
    parser.add_argument("--reco-binned", action="store_true",
                         help="Also plot the single-pass reco-pt-binned SF (jet_correction."
                              "derive_correction) for comparison. Off by default: the "
                              "truth-inverted correction (jet_correction.derive_response_curve "
                              "+ apply_correction_inverted) supersedes it -- closure vs pt/eta is "
                              "several times better, since binning in the true reference pt "
                              "avoids the reco-pt migration bias the binned SF has.")
    parser.add_argument("--fit", action="store_true",
                         help="Also derive and plot the smooth-fit correction (jet_correction."
                              "derive_correction_fit) for comparison. Off by default: it was "
                              "tested and gives essentially the same closure as the single-pass "
                              "binned SF (fitting doesn't fix the migration bias, only smooths "
                              "the same biased input) -- kept as an option, not the default.")
    parser.add_argument("--iterative", action="store_true",
                         help="Also derive and plot the derive-apply-re-derive iterative "
                              "correction (jet_correction.derive_correction_iterative). Off by "
                              "default: tested and it converges to essentially the same result "
                              "as one round (re-deriving the same reco-pt-binned estimator on "
                              "its own output is close to a no-op) -- kept as an option, not the "
                              "default. See --n-iter.")
    parser.add_argument("--n-iter", type=int, default=3,
                         help="Number of rounds for --iterative (default 3)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print("Loading %s" % args.table)
    d = np.load(args.table)
    table = {k: d[k] for k in d.files}
    print("  %d matched jet pairs" % len(table["dr"]))

    pt_bins = np.array(args.pt_bins) if args.pt_bins else None
    eta_bins = np.array(args.eta_bins) if args.eta_bins else None

    corr = jet_correction.derive_correction(table, pt_bins=pt_bins, eta_bins=eta_bins,
                                             min_stat=args.min_stat)
    corr_path = os.path.join(args.outdir, "correction_map.npz")
    jet_correction.save_correction(corr_path, corr)
    print("Wrote correction map to %s (%d x %d bins)"
          % (corr_path, len(corr["eta_bins"]) - 1, len(corr["pt_bins"]) - 1))

    curve = jet_correction.derive_response_curve(table, pt_bins=pt_bins, eta_bins=eta_bins,
                                                  min_stat=args.min_stat)
    curve_path = os.path.join(args.outdir, "response_curve.npz")
    jet_correction.save_response_curve(curve_path, curve)
    print("Wrote truth-binned response curve to %s" % curve_path)

    tables = {
        "raw": table,
        "SF-corrected": _inverted_corrected_table(table, curve),
    }
    if args.reco_binned:
        tables["SF-corrected (reco-binned)"] = _corrected_table(table, corr)
    if args.fit:
        fit_corr = jet_correction.derive_correction_fit(table, eta_bins=eta_bins, min_stat=args.min_stat)
        fit_path = os.path.join(args.outdir, "correction_fit.npz")
        jet_correction.save_correction_fit(fit_path, fit_corr)
        print("Wrote fit correction to %s (%d/%d eta bins fit ok)"
              % (fit_path, int(np.sum(fit_corr["fit_ok"])), len(fit_corr["fit_ok"])))
        tables["SF-corrected (fit)"] = _fit_corrected_table(table, fit_corr)
        _plot_fit_qa(table, fit_corr, args.outdir, args.label)
    if args.iterative:
        corrections = jet_correction.derive_correction_iterative(
            table, n_iter=args.n_iter, pt_bins=pt_bins, eta_bins=eta_bins, min_stat=args.min_stat)
        iter_path = os.path.join(args.outdir, "correction_iterative.npz")
        jet_correction.save_correction_iterative(iter_path, corrections)
        print("Wrote iterative correction (%d rounds) to %s" % (args.n_iter, iter_path))
        tables["SF-corrected (%d rounds)" % args.n_iter] = _iterative_corrected_table(table, corrections)
        _print_iterative_convergence(table, corrections, pt_bins, eta_bins, tuple(args.eta_pt_range), args.label)

    _plot_vs_pt(tables, args.outdir, args.label, pt_bins)
    _plot_vs_eta(tables, args.outdir, args.label, eta_bins, tuple(args.eta_pt_range))
    _plot_correction_maps(corr, args.outdir, args.label)
    _print_closure(tables, pt_bins, eta_bins, tuple(args.eta_pt_range), args.label)

    print("Wrote plots to %s" % args.outdir)


if __name__ == "__main__":
    main()
