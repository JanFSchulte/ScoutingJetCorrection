"""Scouting-vs-offline jet response/resolution helpers.

This is a trimmed copy of the response.py module shared with the (separate)
ScoutingPuppiCalibration/Calibration PUPPI-recalibration package: only the
data/data (scouting-vs-offline, real collisions or MC-side offline-jet
comparison) functions are kept here. The gen-truth-based functions
(response_table, response_vs_pt, response_vs_pu, mass_table,
mass_resolution_vs_pt/pu, fake_rate_vs_pt/pu), which need io.matched_response/
matched_mass and genJetIdx-based matching, are NOT needed by this package and
were deliberately left out rather than copied unused.
"""

import numpy as np


def _binned_stats(x, y, bins):
    """Median and IQR-based resolution of y in bins of x."""
    idx = np.digitize(x, bins) - 1
    centers, medians, resolutions, counts = [], [], [], []
    for b in range(len(bins) - 1):
        sel = idx == b
        n = np.count_nonzero(sel)
        if n < 20:
            continue
        yy = y[sel]
        q16, q50, q84 = np.percentile(yy, [16, 50, 84])
        centers.append(0.5 * (bins[b] + bins[b + 1]))
        medians.append(q50)
        resolutions.append(0.5 * (q84 - q16))  # IQR/2, robust to non-Gaussian tails
        counts.append(n)
    return np.array(centers), np.array(medians), np.array(resolutions), np.array(counts)


def response_vs_pt_data(table, pt_bins=None, eta_max=2.5, min_scout_pt=10.0):
    """Median response + resolution binned in the true (offline) reference
    jet pt, for the scouting-vs-offline comparison table from
    data_matching.build_comparison_table() (scout_pt/scout_eta/off_pt fields,
    dR-matched scouting-vs-offline jet pairs in real collisions or MC's
    offline-jet side -- no gen truth involved). response = scout_pt / off_pt,
    binned in off_pt.

    min_scout_pt drops pairs with scout_pt below this (default 10 GeV --
    deliberately BELOW the 15 GeV hard floor MiniAOD/NanoAOD enforces on
    off_pt, since a genuine raw scouting jet at 10-14 GeV can have a real,
    if large, negative response and legitimately belongs in this
    diagnostic's population; only the most extreme, near-zero-pt tail is
    excluded). The scouting side has no floor of its own, so
    dr_match_jets()'s purely geometric matching can pair a real >=15 GeV
    offline jet with an unrelated near-zero-pt scouting noise candidate when
    the genuine scouting jet wasn't reconstructed -- a matching/efficiency
    failure, not a response difference. Checked empirically: dR to the
    matched offline jet degrades smoothly as scout_pt drops to 0, with no
    sharp "noise vs. real" elbow, so this threshold is a judgment call, not
    a uniquely correct number -- see jet_correction.derive_response_curve's
    docstring for the full rationale (same issue on the correction-
    derivation side). Pass None to disable.

    The selection is always evaluated against table["scout_pt_raw"] if that
    key is present, falling back to table["scout_pt"] otherwise -- NOT
    always table["scout_pt"] directly. This matters for "raw vs corrected"
    closure comparisons: every _corrected()-style helper across this repo
    overwrites table["scout_pt"] in place with the SF-corrected value, so
    filtering on "scout_pt" there would select a DIFFERENT set of jets for
    the corrected curve than for the raw one (some jets cross the threshold
    in either direction under correction) -- comparing bias before/after on
    two different populations, not a real closure check. scout_pt_raw is
    set once, before any correction is applied, and never overwritten, so
    the same jets are compared throughout.
    """
    if pt_bins is None:
        pt_bins = np.array([15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 300, 500])
    scout_pt_for_sel = table.get("scout_pt_raw", table["scout_pt"])
    sel = np.abs(table["scout_eta"]) < eta_max
    if min_scout_pt is not None:
        sel &= scout_pt_for_sel >= min_scout_pt
    response = table["scout_pt"][sel] / table["off_pt"][sel]
    return _binned_stats(table["off_pt"][sel], response, pt_bins)


def response_vs_eta_data(table, eta_bins=None, pt_range=(30.0, None), min_scout_pt=10.0):
    """Same population as response_vs_pt_data, but binned in scout_eta instead
    of off_pt, at a fixed off_pt window (pt_range) so the strong pt-dependence
    of response doesn't leak into the eta-dependence being studied.

    min_scout_pt: see response_vs_pt_data's docstring, including why the
    selection uses table["scout_pt_raw"] (falling back to table["scout_pt"])
    rather than table["scout_pt"] directly. Default pt_range already starts
    at 30 GeV (above where the offline/scouting pt-floor mismatch matters),
    so this rarely changes anything here in practice -- kept for consistency
    and for callers that pass a lower pt_range.
    """
    if eta_bins is None:
        # Matches jet_correction._default_eta_bins(): core |eta|<2.5 in 0.5-wide
        # bins, plus two forward bins per side out to 3.0 -- see that function's
        # comment for why the forward edge sits at 3.0, not further.
        core = np.linspace(-2.5, 2.5, 11)
        eta_bins = np.concatenate(([-3.0, -2.7], core, [2.7, 3.0]))
    scout_pt_for_sel = table.get("scout_pt_raw", table["scout_pt"])
    pt_lo, pt_hi = pt_range
    sel = np.ones(len(table["off_pt"]), dtype=bool)
    if pt_lo is not None:
        sel &= table["off_pt"] >= pt_lo
    if pt_hi is not None:
        sel &= table["off_pt"] < pt_hi
    if min_scout_pt is not None:
        sel &= scout_pt_for_sel >= min_scout_pt
    response = table["scout_pt"][sel] / table["off_pt"][sel]
    return _binned_stats(table["scout_eta"][sel], response, eta_bins)
