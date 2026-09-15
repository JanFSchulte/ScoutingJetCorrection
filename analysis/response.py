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


def response_vs_pt_data(table, pt_bins=None, eta_max=2.5, min_scout_pt=15.0):
    """Median response + resolution binned in the true (offline) reference
    jet pt, for the scouting-vs-offline comparison table from
    data_matching.build_comparison_table() (scout_pt/scout_eta/off_pt fields,
    dR-matched scouting-vs-offline jet pairs in real collisions or MC's
    offline-jet side -- no gen truth involved). response = scout_pt / off_pt,
    binned in off_pt.

    min_scout_pt drops pairs with scout_pt below this (default 15 GeV,
    matching the hard floor MiniAOD/NanoAOD already enforces on off_pt --
    off_pt.min() is exactly 15.0 in every comparison table checked). The
    scouting side has no such floor, so dr_match_jets()'s purely geometric
    matching can pair a real >=15 GeV offline jet with an unrelated near-
    zero-pt scouting noise candidate when the genuine scouting jet wasn't
    reconstructed -- a matching/efficiency failure, not a response
    difference, and left unfiltered it visibly biases the lowest pt bin (up
    to ~22% of MC pairs, ~10% of data pairs, in off_pt in [15,20) -- see
    jet_correction.derive_response_curve's docstring for the same issue on
    the correction-derivation side). Pass None to disable.
    """
    if pt_bins is None:
        pt_bins = np.array([15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 300, 500])
    sel = np.abs(table["scout_eta"]) < eta_max
    if min_scout_pt is not None:
        sel &= table["scout_pt"] >= min_scout_pt
    response = table["scout_pt"][sel] / table["off_pt"][sel]
    return _binned_stats(table["off_pt"][sel], response, pt_bins)


def response_vs_eta_data(table, eta_bins=None, pt_range=(30.0, None), min_scout_pt=15.0):
    """Same population as response_vs_pt_data, but binned in scout_eta instead
    of off_pt, at a fixed off_pt window (pt_range) so the strong pt-dependence
    of response doesn't leak into the eta-dependence being studied.

    min_scout_pt: see response_vs_pt_data's docstring. Default pt_range
    already starts at 30 GeV (above where the offline/scouting pt-floor
    mismatch matters), so this rarely changes anything here in practice --
    kept for consistency and for callers that pass a lower pt_range.
    """
    if eta_bins is None:
        # Matches jet_correction._default_eta_bins(): core |eta|<2.5 in 0.5-wide
        # bins, plus two forward bins per side out to 3.0 -- see that function's
        # comment for why the forward edge sits at 3.0, not further.
        core = np.linspace(-2.5, 2.5, 11)
        eta_bins = np.concatenate(([-3.0, -2.7], core, [2.7, 3.0]))
    pt_lo, pt_hi = pt_range
    sel = np.ones(len(table["off_pt"]), dtype=bool)
    if pt_lo is not None:
        sel &= table["off_pt"] >= pt_lo
    if pt_hi is not None:
        sel &= table["off_pt"] < pt_hi
    if min_scout_pt is not None:
        sel &= table["scout_pt"] >= min_scout_pt
    response = table["scout_pt"][sel] / table["off_pt"][sel]
    return _binned_stats(table["scout_eta"][sel], response, eta_bins)
