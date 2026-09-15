"""Package jet_correction.py's derive_response_curve() maps into
correctionlib (https://github.com/cms-nanoAOD/correctionlib) Correction
objects, so the same pt/eta-dependent scouting-vs-offline calibration can be
looked up from coffea (via correctionlib's python/awkward bindings) or
RDataFrame (via its C++ API) without depending on this analysis package.

Why a MultiBinning and not something that reproduces derive_response_curve's
own machinery exactly: derive_response_curve()/apply_correction_inverted()
work in two steps -- (1) bin the TRUE reference pt (off_pt) to build
R(true_pt, eta) = median(scout_pt/off_pt) free of reco-pt migration bias, (2)
invert it per-jet by fixed-point iteration, since only the RAW (reco) pt is
available at application time. correctionlib's node types (Binning,
MultiBinning, Category, Formula, ...) are direct lookups/formulas -- there is
no node that runs an iterative solve. So step (2) is done ONCE here, ahead of
time, on a fine grid of raw pt: for each (eta, raw_pt) grid cell,
apply_correction_inverted() gives the same corrected pt this analysis code
would produce for a jet born at that (raw_pt, eta), and the cell's SF is
corrected_pt / raw_pt. This is exactly how standard JEC corrections are
delivered too (a lookup/formula keyed on RAW jet pt and eta) -- baking a
one-time numerical solve into a delivered pt-dependent table/formula is
standard practice, not a shortcut specific to this analysis.

The grid uses n_pt_eval log-spaced pt bins (finer than derive_response_curve's
own ~12 pt_bins) so the baked step-function lookup tracks the original
piecewise-LINEAR (np.interp) inversion closely; eta bins are reused as-is
from the source curve, since jet_correction._default_eta_bins() already
encodes physically-motivated bin edges (see that function's docstring), not
an arbitrary resolution choice.
"""

import numpy as np
import correctionlib.schemav2 as cs

import jet_correction

# Fixed, generous pt range for the baked fine grid -- deliberately NOT each
# curve's own pt_bins[0]/[-1]. Some curves are derived over a much narrower
# range than "any raw jet pt a user might query" (e.g. the AK8 b-tag data
# category only has calibrated bins from 200-1000 GeV, since boosted/AK8
# b-tagged jets don't exist below ~200 GeV) -- apply_correction_inverted()
# already extrapolates correctly beyond ITS OWN pt_bins-derived boundary for
# a query outside that curve's range (flat SF from the boundary bin, applied
# multiplicatively -- NOT a constant output pt; see that function's
# docstring for the cutoff bug this replaced), but only if the query point
# is actually sampled when building the grid. Restricting the baked grid to
# [pt_bins[0], pt_bins[-1]] meant queries further outside that hit
# correctionlib's flow="clamp" instead (repeats the nearest FINE BIN's
# content, computed for a pt near that curve's own boundary) -- silently
# disagreeing with what apply_correction_inverted would say for the same
# query. Sampling a shared, wide range for every curve makes the baked table
# agree with apply_correction_inverted's real (already-correct) extrapolation
# everywhere a real jet could plausibly land; only genuinely extreme queries
# outside this range fall back to correctionlib's own flow="clamp".
PT_EVAL_RANGE = (10.0, 2000.0)


def response_curve_to_multibinning(curve, n_pt_eval=60,
                                    eta_var="JetEta", pt_var="JetPt"):
    """Bake a derive_response_curve() map (defined in TRUE pt) into a
    correctionlib MultiBinning keyed on RAW (reco) pt and eta -- see module
    docstring. Returns a correctionlib.schemav2.MultiBinning whose evaluate()
    gives the multiplicative SF to apply to a raw jet's pt."""
    eta_edges = np.asarray(curve["eta_bins"], dtype=float)
    pt_lo, pt_hi = PT_EVAL_RANGE
    pt_edges = np.geomspace(pt_lo, pt_hi, n_pt_eval + 1)

    eta_centers = 0.5 * (eta_edges[:-1] + eta_edges[1:])
    pt_centers = np.sqrt(pt_edges[:-1] * pt_edges[1:])  # geometric mean, log-spaced bins

    n_eta = len(eta_centers)
    content = np.empty((n_eta, n_pt_eval))
    for ie, eta_c in enumerate(eta_centers):
        eta_arr = np.full(n_pt_eval, eta_c)
        corrected_pt = jet_correction.apply_correction_inverted(pt_centers, eta_arr, curve)
        content[ie] = corrected_pt / pt_centers

    return cs.MultiBinning(
        nodetype="multibinning",
        inputs=[eta_var, pt_var],
        edges=[eta_edges.tolist(), pt_edges.tolist()],
        content=content.flatten().tolist(),  # eta slowest-varying, pt fastest -- matches inputs order
        flow="clamp",  # jets outside the calibrated range get the nearest edge bin's SF, not extrapolation
    )


def build_correction(name, description, curves_by_category, n_pt_eval=60,
                      category_var="systematic", eta_var="JetEta", pt_var="JetPt"):
    """curves_by_category: {category_name: curve_dict}, must include
    "inclusive". No `default` is set on the Category node -- an unrecognized
    category string raises immediately instead of silently falling back to
    some other map, since a silent wrong-category lookup in a calibration SF
    is worse than a crash."""
    if "inclusive" not in curves_by_category:
        raise ValueError("curves_by_category must include an 'inclusive' entry")

    items = [
        cs.CategoryItem(key=cat_name, value=response_curve_to_multibinning(
            curve, n_pt_eval=n_pt_eval, eta_var=eta_var, pt_var=pt_var))
        for cat_name, curve in curves_by_category.items()
    ]
    data = cs.Category(nodetype="category", input=category_var, content=items)

    return cs.Correction(
        name=name,
        description=description,
        version=1,
        inputs=[
            cs.Variable(name=category_var, type="string",
                        description="Category the response map was derived for: %s"
                                     % ", ".join(sorted(curves_by_category))),
            cs.Variable(name=eta_var, type="real", description="Raw scouting jet eta"),
            cs.Variable(name=pt_var, type="real", description="Raw scouting jet pt [GeV]"),
        ],
        output=cs.Variable(name="correction", type="real",
                            description="Multiplicative factor: corrected_pt = correction * raw_pt"),
        data=data,
    )
