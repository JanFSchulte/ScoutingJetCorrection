"""pT/eta-binned scale-factor correction, derived from a scout-vs-offline
matched comparison table (data_matching.build_comparison_table or
build_comparison_table_mc -- both produce the same scout_pt/scout_eta/off_pt
shape, so this module works on either).

Binning is done in the RAW scouting jet's own (pt, eta) -- not off_pt/gen_pt --
because that's the only information available on a jet at correction-
application time (no matched offline/gen jet exists for jets outside this
closure sample). This mirrors how standard JEC L2L3(Residual) corrections are
derived and applied: as a function of raw reco jet kinematics, looked up from
a map built once on a well-matched calibration sample.

SF(pt, eta) = median(off_pt / scout_pt) in each bin -- removes the pt/eta-
dependent response bias. The residual resolution (IQR/2 of the scale-
corrected response) is also reported per bin, purely as a diagnostic of how
much spread SF cannot remove -- it is NOT applied to jets. A deterministic
scale factor can only shift a bin's median; it cannot narrow the underlying
stochastic spread, so there is no "smearing" step here (an earlier version
of this module applied Gaussian smearing sized to that residual, but
smearing can only *widen* a distribution, never fix it -- removed).
"""

import numpy as np


def _default_pt_bins():
    return np.array([15, 20, 25, 30, 40, 50, 70, 100, 150, 200, 300, 500, 800])


def _default_eta_bins():
    # Core |eta|<2.5 region in the usual 0.5-wide bins, plus two forward bins
    # per side out to 3.0 (2.5-2.7, 2.7-3.0). Jets out there are real (both
    # matched samples have entries to |eta|~3.0), but the scouting online PF
    # reconstruction has a hard acceptance edge around |eta|~3.0 (HLTScoutingPF
    # Producer.cc's default pfJetEtaCut=3.0, and an artificial pileup of jets
    # in ~2.65-2.9 from clustering against that truncated candidate coverage)
    # -- so a correction out here is provided for completeness, not because
    # it's expected to be trustworthy or used. See jet_correction module
    # discussion / project memory for how this boundary was found.
    core = np.linspace(-2.5, 2.5, 11)
    return np.concatenate(([-3.0, -2.7], core, [2.7, 3.0]))


def derive_correction(table, pt_bins=None, eta_bins=None,
                       pt_field="scout_pt", eta_field="scout_eta", ref_field="off_pt",
                       min_stat=50):
    """Build the 2D (eta, pt) scale-factor map.

    Returns a dict: pt_bins, eta_bins (bin edges), sf, resolution, n (each
    [n_eta_bins, n_pt_bins] arrays; resolution is the post-SF IQR/2, a
    diagnostic only). Bins with fewer than min_stat entries get sf=1,
    resolution=0 (no-op) -- not enough statistics to trust a correction
    there, and downstream lookups still work as clipped no-ops.
    """
    if pt_bins is None:
        pt_bins = _default_pt_bins()
    if eta_bins is None:
        eta_bins = _default_eta_bins()
    pt_bins = np.asarray(pt_bins, dtype=float)
    eta_bins = np.asarray(eta_bins, dtype=float)

    pt = table[pt_field]
    eta = table[eta_field]
    ref = table[ref_field]
    response = pt / ref

    n_eta, n_pt = len(eta_bins) - 1, len(pt_bins) - 1
    sf = np.ones((n_eta, n_pt))
    resolution = np.zeros((n_eta, n_pt))
    n = np.zeros((n_eta, n_pt), dtype=int)

    ipt = np.digitize(pt, pt_bins) - 1
    ieta = np.digitize(eta, eta_bins) - 1

    for ie in range(n_eta):
        for ip in range(n_pt):
            sel = (ipt == ip) & (ieta == ie)
            count = int(np.count_nonzero(sel))
            n[ie, ip] = count
            if count < min_stat:
                continue
            median_response = np.median(response[sel])
            sf_bin = 1.0 / median_response
            corrected_response = response[sel] * sf_bin
            q16, q84 = np.percentile(corrected_response, [16, 84])
            sf[ie, ip] = sf_bin
            resolution[ie, ip] = 0.5 * (q84 - q16)

    return {"pt_bins": pt_bins, "eta_bins": eta_bins, "sf": sf, "resolution": resolution, "n": n}


def _bin_index(values, edges):
    """Clip to the valid bin range instead of extrapolating outside it."""
    idx = np.digitize(values, edges) - 1
    return np.clip(idx, 0, len(edges) - 2)


def apply_correction(pt, eta, corr):
    """Look up SF(pt, eta) from a derive_correction() map and apply it to raw
    (pt, eta) arrays. Returns the corrected pt array; input is untouched."""
    ipt = _bin_index(pt, corr["pt_bins"])
    ieta = _bin_index(eta, corr["eta_bins"])
    sf = corr["sf"][ieta, ipt]
    return np.asarray(pt) * sf


def save_correction(path, corr):
    np.savez(path, pt_bins=corr["pt_bins"], eta_bins=corr["eta_bins"],
              sf=corr["sf"], resolution=corr["resolution"], n=corr["n"])


def load_correction(path):
    d = np.load(path)
    return {"pt_bins": d["pt_bins"], "eta_bins": d["eta_bins"],
            "sf": d["sf"], "resolution": d["resolution"], "n": d["n"]}


# ---------------------------------------------------------------------------
# Iterative variant: the single-pass SF map above is derived in bins of raw
# scout_pt but evaluated (in the response-vs-off_pt plots) in bins of off_pt.
# With a steeply falling jet spectrum and finite resolution, a given raw-pt
# bin is a resolution-skewed mix of true pt values, so a bin's median SF
# doesn't fully invert that migration -- confirmed empirically: neither
# finer binning nor a smooth fit (both still derived from the same raw-pt-
# binned medians) closes this gap.
#
# Fix: re-derive the SF map on the ALREADY-CORRECTED pt from the previous
# round, and repeat. Each round's correction is derived from a pt estimate
# that is closer to true pt than the round before, so the reco-vs-truth
# migration shrinks round over round -- standard practice for deriving
# L2L3Residual-style JEC chains. Unlike the single map above, this does NOT
# collapse into one 2D grid: round k's SF is a function of round (k-1)'s
# corrected pt, not the original raw pt, so applying it means walking the
# chain of maps in order.
# ---------------------------------------------------------------------------

def derive_correction_iterative(table, n_iter=3, pt_bins=None, eta_bins=None,
                                 pt_field="scout_pt", eta_field="scout_eta", ref_field="off_pt",
                                 min_stat=50):
    """Returns a list of n_iter correction dicts (each shaped like
    derive_correction()'s output). Round 0 is derived on the raw pt exactly
    like derive_correction(); round k>0 is derived on the pt already
    corrected by rounds 0..k-1. apply_correction_iterative() applies all of
    them in sequence to a fresh (pt, eta) pair.
    """
    eta = np.asarray(table[eta_field])
    ref = np.asarray(table[ref_field])
    current_pt = np.array(table[pt_field], dtype=float)

    corrections = []
    for _ in range(n_iter):
        iter_table = {pt_field: current_pt, eta_field: eta, ref_field: ref}
        corr = derive_correction(iter_table, pt_bins=pt_bins, eta_bins=eta_bins,
                                  pt_field=pt_field, eta_field=eta_field, ref_field=ref_field,
                                  min_stat=min_stat)
        current_pt = apply_correction(current_pt, eta, corr)
        corrections.append(corr)
    return corrections


def apply_correction_iterative(pt, eta, corrections):
    """Apply a derive_correction_iterative() chain in sequence. Returns the
    corrected pt array; input is untouched."""
    pt = np.asarray(pt, dtype=float)
    for corr in corrections:
        pt = apply_correction(pt, eta, corr)
    return pt


def save_correction_iterative(path, corrections):
    np.savez(path, n_iter=len(corrections),
              pt_bins=corrections[0]["pt_bins"], eta_bins=corrections[0]["eta_bins"],
              sf=np.stack([c["sf"] for c in corrections]),
              resolution=np.stack([c["resolution"] for c in corrections]),
              n=np.stack([c["n"] for c in corrections]))


def load_correction_iterative(path):
    d = np.load(path)
    n_iter = int(d["n_iter"])
    return [{"pt_bins": d["pt_bins"], "eta_bins": d["eta_bins"],
              "sf": d["sf"][i], "resolution": d["resolution"][i], "n": d["n"][i]}
             for i in range(n_iter)]


# ---------------------------------------------------------------------------
# Truth-binned inversion: unlike everything above (all functions of raw
# scout_pt, all showing the same handful-of-percent low-pt non-closure when
# evaluated against off_pt bins -- confirmed empirically to persist across
# finer binning, a smooth fit, and repeated re-derivation of the same
# estimator), this derives the response curve
#     R(true_pt, eta) = median(scout_pt / off_pt)
# binned in the TRUE reference pt (off_pt), not raw scout_pt. That binning
# has no migration bias by construction: conditioning on the true pt means
# there's no spectrum-driven population mixing the way there is when
# conditioning on reco pt (a falling spectrum + finite resolution biases
# which true-pt jets populate a given RECO-pt bin; it can't bias which
# jets populate a given TRUE-pt bin, since that's the conditioning variable
# itself). This is exactly what response_vs_pt_data/response_vs_eta_data
# already compute -- this just does it on a 2D (eta, pt) grid so it can be
# inverted per-jet.
#
# To correct a raw jet (whose true pt is unknown), solve the implicit
# equation raw_pt = true_pt * R(true_pt, eta) for true_pt by fixed-point
# iteration: true_pt_{k+1} = raw_pt / R(true_pt_k, eta), starting from
# true_pt_0 = raw_pt. R is evaluated by linear interpolation in pt (per eta
# bin) rather than a bin lookup, so the iteration isn't destabilized by
# bin-edge steps.
# ---------------------------------------------------------------------------

def derive_response_curve(table, pt_bins=None, eta_bins=None,
                           pt_field="scout_pt", eta_field="scout_eta", ref_field="off_pt",
                           min_stat=50):
    """R(true_pt, eta) = median(scout_pt/off_pt), binned in ref_field (the
    true reference pt) and eta_field. Returns pt_bins, eta_bins, r, n
    ([n_eta_bins, n_pt_bins] arrays; r defaults to 1, a no-op, in bins with
    fewer than min_stat entries)."""
    if pt_bins is None:
        pt_bins = _default_pt_bins()
    if eta_bins is None:
        eta_bins = _default_eta_bins()
    pt_bins = np.asarray(pt_bins, dtype=float)
    eta_bins = np.asarray(eta_bins, dtype=float)

    true_pt = table[ref_field]
    eta = table[eta_field]
    response = table[pt_field] / table[ref_field]

    n_eta, n_pt = len(eta_bins) - 1, len(pt_bins) - 1
    r = np.ones((n_eta, n_pt))
    n = np.zeros((n_eta, n_pt), dtype=int)

    ipt = np.digitize(true_pt, pt_bins) - 1
    ieta = np.digitize(eta, eta_bins) - 1

    for ie in range(n_eta):
        for ip in range(n_pt):
            sel = (ipt == ip) & (ieta == ie)
            count = int(np.count_nonzero(sel))
            n[ie, ip] = count
            if count < min_stat:
                continue
            r[ie, ip] = np.median(response[sel])

    return {"pt_bins": pt_bins, "eta_bins": eta_bins, "r": r, "n": n}


def derive_response_curve_by_category(table, categories, category_field="off_hadronFlavour",
                                       **kwargs):
    """Derive a separate derive_response_curve() map per named jet category
    (e.g. flavour), so that jets of a given category can be corrected with a
    map derived ONLY from that category, rather than one inclusive map
    averaged over a mixed population. Motivating case: an inclusive AK4
    correction (fit mostly to light/gluon jets, the majority species) left a
    real ~4% residual pt bias on b-jets specifically when checked against a
    HH->bbtautau MC sample -- a dedicated b-jet map is meant to close that.

    categories: {name: predicate}, where predicate(table[category_field]) ->
    boolean mask, e.g. {"b": lambda f: f == 5, "light": lambda f: f == 0}.
    kwargs are forwarded to derive_response_curve (pt_bins, eta_bins, etc).

    Returns {name: curve}. Categories with too few entries for any bin to
    reach min_stat still return a curve (all-1 no-op, from
    derive_response_curve's own min_stat handling) rather than raising.
    """
    field = table[category_field]
    curves = {}
    for name, predicate in categories.items():
        mask = predicate(field)
        sub = {k: v[mask] for k, v in table.items()}
        curves[name] = derive_response_curve(sub, **kwargs)
    return curves


def apply_correction_inverted(pt, eta, curve, tol=1e-3, max_iter=60):
    """Solve raw_pt = true_pt * R(true_pt, eta) for true_pt, using a
    derive_response_curve() map. R(true_pt, eta) itself is clamped to the
    curve's pt-bin-center range (flat extrapolation of the SCALE FACTOR
    beyond the calibrated range -- matches apply_correction's clip-don't-
    extrapolate-the-SF policy), but the solved true_pt is NOT clamped: a raw
    jet pt outside the calibrated range gets true_pt = raw_pt / R(pt_lo or
    pt_hi, eta), the same flat SF applied multiplicatively, not pinned to a
    constant. Returns the corrected (estimated true) pt array; input is
    untouched.

    An earlier version of this function clamped the OUTPUT true_pt itself to
    [pt_lo, pt_hi] for any raw_pt outside the achievable range -- e.g. for a
    curve whose last pt bin is [500, 800) GeV (pt_hi = 650, the bin center),
    EVERY raw jet pt above ~650-700 GeV (wherever g_hi_full landed) was
    corrected to the exact same constant ~650 GeV, a hard cutoff. Silent and
    severe for any analysis with jets above that range (found via the baked
    correctionlib export, whose fine pt grid runs to 2000 GeV -- see
    correctionlib_export.PT_EVAL_RANGE). Fixed by keeping the SF clamp (needed
    since R has no data outside the calibrated bins) but no longer clamping
    the pt this SF is applied to.

    Solved by BISECTION on true_pt in [pt_lo, pt_hi] for raw_pt within the
    achievable range, not fixed-point iteration (true_pt <- raw_pt/R(true_pt),
    the original implementation here) -- that iteration can OSCILLATE
    INDEFINITELY instead of converging wherever R(pt) has a steep or noisy
    bin-to-bin transition (confirmed on real curves, e.g. AK8 b-jets around
    pt~60-175 GeV, where per-bin medians jump R: 1.0 -> 2.11 -> 1.36 -> 0.998
    -- the map's local derivative exceeds 1 there, so it's not a
    contraction), silently returning whatever arbitrary, iteration-count-
    dependent value the loop happened to be on at cutoff. A systematic sweep
    across all six derived correction groups (see build_correctionlib_set.py's
    validation) found this affecting a good fraction of every group to some
    degree (mild, ~10% of the pt/eta grid with a several-percent shift, for
    AK4) and severely for AK8 MC (up to 47% relative swings between iteration
    counts -- i.e. genuinely divergent, not just slow).

    Bisection cannot diverge: each step halves the bracket regardless of R's
    local behavior, so it always converges (to tol) to A root of
    true_pt*R(true_pt) = raw_pt within [pt_lo, pt_hi], assuming that product
    is roughly monotonic increasing overall (true for a physical
    reco-pt-vs-true-pt response, even where R itself dips locally). raw pt
    values outside the achievable range [pt_lo*R(pt_lo), pt_hi*R(pt_hi)] are
    inverted directly (closed form, R is constant out there) rather than by
    bisection.
    """
    pt = np.asarray(pt, dtype=float)
    eta = np.asarray(eta, dtype=float)
    ieta = _bin_index(eta, curve["eta_bins"])
    pt_centers = 0.5 * (curve["pt_bins"][:-1] + curve["pt_bins"][1:])
    pt_lo, pt_hi = pt_centers[0], pt_centers[-1]
    r_matrix = curve["r"]
    n_eta = r_matrix.shape[0]

    def r_lookup(true_pt, ieta_sub):
        clamped = np.clip(true_pt, pt_lo, pt_hi)
        r_vals = np.empty_like(true_pt)
        for ie in range(n_eta):
            sel = ieta_sub == ie
            if np.any(sel):
                r_vals[sel] = np.interp(clamped[sel], pt_centers, r_matrix[ie])
        return r_vals

    def predicted_raw(true_pt, ieta_sub):
        return true_pt * r_lookup(true_pt, ieta_sub)

    result = np.empty_like(pt)
    g_lo_full = predicted_raw(np.full_like(pt, pt_lo), ieta)
    g_hi_full = predicted_raw(np.full_like(pt, pt_hi), ieta)

    below = pt <= g_lo_full
    above = pt >= g_hi_full
    # R is flat (clamped) beyond pt_lo/pt_hi, so raw_pt = true_pt * R(pt_lo
    # or pt_hi) is linear in true_pt out there -- invert it directly instead
    # of pinning the output to the boundary (see docstring).
    if np.any(below):
        result[below] = pt[below] / r_lookup(np.full(int(below.sum()), pt_lo), ieta[below])
    if np.any(above):
        result[above] = pt[above] / r_lookup(np.full(int(above.sum()), pt_hi), ieta[above])

    solve = ~(below | above)
    if np.any(solve):
        lo = np.full(int(solve.sum()), pt_lo)
        hi = np.full(int(solve.sum()), pt_hi)
        target = pt[solve]
        ieta_s = ieta[solve]
        for _ in range(max_iter):
            mid = 0.5 * (lo + hi)
            g_mid = predicted_raw(mid, ieta_s)
            below_target = g_mid < target
            lo = np.where(below_target, mid, lo)
            hi = np.where(below_target, hi, mid)
            if np.max(hi - lo) < tol:
                break
        result[solve] = 0.5 * (lo + hi)

    return result


def save_response_curve(path, curve):
    np.savez(path, pt_bins=curve["pt_bins"], eta_bins=curve["eta_bins"],
              r=curve["r"], n=curve["n"])


def load_response_curve(path):
    d = np.load(path)
    return {"pt_bins": d["pt_bins"], "eta_bins": d["eta_bins"], "r": d["r"], "n": d["n"]}


# ---------------------------------------------------------------------------
# Smooth-fit variant: same inputs, same derivation logic (median response of
# scout_pt/off_pt in bins of raw scout pt/eta), but instead of taking each
# bin's median as a flat step-function SF, fits a smooth function of pt to
# the binned medians in each eta slice -- the standard CMS JEC L2Relative
# approach (response(pt) = p0 + p1/pt + p2/pt^2 + p3*ln(pt), fit per eta
# bin). This removes bin-edge artifacts and lets the correction extrapolate
# smoothly instead of clamping to one edge bin's value.
#
# IMPORTANT: the fit is performed on medians computed in bins of the SAME
# raw scout_pt used by derive_correction, so it inherits the identical
# reco-pt-vs-true-pt migration bias described in the module docstring above
# -- smoothing the SF curve does not remove that structural non-closure, it
# only removes binning noise/edge artifacts. Confirmed empirically: fit and
# binned-median closure are essentially the same at low pt where the
# migration bias dominates (see run_offline_comparison_plots.py's printed
# closure comparison).
# ---------------------------------------------------------------------------

def _l2rel_func(pt, p0, p1, p2, p3):
    return p0 + p1 / pt + p2 / pt ** 2 + p3 * np.log(pt)


def _median_vs_pt(pt, response, pt_bins, min_stat):
    idx = np.digitize(pt, pt_bins) - 1
    centers, medians, counts = [], [], []
    for b in range(len(pt_bins) - 1):
        sel = idx == b
        n = int(np.count_nonzero(sel))
        if n < min_stat:
            continue
        centers.append(0.5 * (pt_bins[b] + pt_bins[b + 1]))
        medians.append(np.median(response[sel]))
        counts.append(n)
    return np.array(centers), np.array(medians), np.array(counts)


def derive_correction_fit(table, eta_bins=None, pt_fit_bins=None,
                           pt_field="scout_pt", eta_field="scout_eta", ref_field="off_pt",
                           min_stat=50, fit_min_stat=20):
    """Smooth-fit alternative to derive_correction(). Returns a dict:
    eta_bins, params ([n_eta_bins, 4] fit coefficients for _l2rel_func),
    fit_ok ([n_eta_bins] bool), pt_fit_range (min/max pt the fit was
    performed over -- apply_correction_fit clamps evaluation to this range
    rather than extrapolating the 1/pt, 1/pt^2 terms into unfit territory).

    eta bins with fewer than min_stat entries, or where curve_fit fails
    (e.g. too few populated pt slices), get params=[1,0,0,0] (SF=1, no-op)
    and fit_ok=False.
    """
    from scipy.optimize import curve_fit

    if eta_bins is None:
        eta_bins = _default_eta_bins()
    if pt_fit_bins is None:
        pt_fit_bins = np.geomspace(15, 800, 41)
    eta_bins = np.asarray(eta_bins, dtype=float)
    pt_fit_bins = np.asarray(pt_fit_bins, dtype=float)

    pt = table[pt_field]
    eta = table[eta_field]
    ref = table[ref_field]
    response = pt / ref

    n_eta = len(eta_bins) - 1
    params = np.zeros((n_eta, 4))
    params[:, 0] = 1.0
    fit_ok = np.zeros(n_eta, dtype=bool)

    ieta = np.digitize(eta, eta_bins) - 1

    for ie in range(n_eta):
        sel_eta = ieta == ie
        if np.count_nonzero(sel_eta) < min_stat:
            continue
        centers, medians, counts = _median_vs_pt(pt[sel_eta], response[sel_eta],
                                                   pt_fit_bins, fit_min_stat)
        if len(centers) < 5:
            continue
        try:
            popt, _ = curve_fit(_l2rel_func, centers, medians, p0=[1.0, 0.0, 0.0, 0.0],
                                 sigma=1.0 / np.sqrt(counts), maxfev=10000)
        except RuntimeError:
            continue
        params[ie] = popt
        fit_ok[ie] = True

    return {"eta_bins": eta_bins, "params": params, "fit_ok": fit_ok,
            "pt_fit_range": np.array([pt_fit_bins[0], pt_fit_bins[-1]])}


def apply_correction_fit(pt, eta, fit_corr):
    """Look up SF(pt, eta) from a derive_correction_fit() map and apply it.
    Evaluation pt is clamped to pt_fit_range -- the 1/pt, 1/pt^2 terms are
    only trustworthy inside the range they were fit over."""
    pt = np.asarray(pt, dtype=float)
    eta = np.asarray(eta, dtype=float)
    ieta = _bin_index(eta, fit_corr["eta_bins"])
    pt_lo, pt_hi = fit_corr["pt_fit_range"]
    pt_eval = np.clip(pt, pt_lo, pt_hi)
    p = fit_corr["params"][ieta]
    predicted_response = _l2rel_func(pt_eval, p[:, 0], p[:, 1], p[:, 2], p[:, 3])
    return pt / predicted_response


def save_correction_fit(path, fit_corr):
    np.savez(path, eta_bins=fit_corr["eta_bins"], params=fit_corr["params"],
              fit_ok=fit_corr["fit_ok"], pt_fit_range=fit_corr["pt_fit_range"])


def load_correction_fit(path):
    d = np.load(path)
    return {"eta_bins": d["eta_bins"], "params": d["params"],
            "fit_ok": d["fit_ok"], "pt_fit_range": d["pt_fit_range"]}
