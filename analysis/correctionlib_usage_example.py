#!/usr/bin/env python3
"""How to apply results/scoutingPUPPI_corrections.json.gz (built by
build_correctionlib_set.py) in a coffea/awkward analysis. Each Correction in
the set takes (systematic: string category, JetEta: real, JetPt: real -- the
RAW/uncorrected scouting jet's own pt) and returns a multiplicative SF:
    corrected_pt = SF * raw_pt

Categories available per Correction (see cset[name].description, or just
cset[name].inputs[0].description): "inclusive" always exists; MC groups
(AK4_plain_MC, AK4_CHS_MC, AK8_MC) additionally have "b"/"light"
(off_hadronFlavour) and "tau"/"nontau" (gen-matched hadronic tau); data
groups (*_Data2024, *_Data2025C_ref) additionally have "btag"/"nobtag"
(offline tagger score >= 0.5). Passing an unrecognized category string
raises immediately rather than silently falling back to something else.

Data2024 (2024G+H+I combined) is the primary data-derived correction, since
2024 is the run period currently used for analysis -- see
build_correctionlib_set.py's GROUPS for why 2024G/H/I were combined (checked
mutually consistent in scale/resolution first) while 2025C was kept as a
SEPARATE *_Data2025C_ref group instead of also being folded in (it showed a
systematic scale/resolution offset from 2024, see plot_era_stability.py).

For RDataFrame (C++), see the docstring at the bottom of this file instead
-- correctionlib ships a C++ header (correction.h) with the same
CorrectionSet/Correction API, usable directly from a Define() call.
"""

import awkward as ak
import numpy as np
import correctionlib

CORRECTIONS_FILE = "results/scoutingPUPPI_corrections.json.gz"


def apply_ak4_plain_mc_correction(jet_pt, jet_eta, hadron_flavour, gentau_dr):
    """Example: dedicated per-category correction for AK4 plain
    (ScoutingPFJetRecluster2) jets in MC, given per-jet raw pt/eta plus the
    truth info needed to pick a category. jet_pt/jet_eta/hadron_flavour/
    gentau_dr are same-shape awkward or numpy arrays (jagged is fine --
    flatten before calling, unflatten after). correctionlib's vectorized
    evaluate() only accepts array arguments for the int/real inputs (eta,
    pt), not the string category, so jets are split into per-category masks
    and evaluate() is called once per category rather than once per jet.
    """
    cset = correctionlib.CorrectionSet.from_file(CORRECTIONS_FILE)
    corr = cset["AK4_plain_MC"]

    flat_pt = np.asarray(ak.flatten(jet_pt))
    counts = ak.num(jet_pt)
    flat_eta = np.asarray(ak.flatten(jet_eta))
    flat_flavour = np.asarray(ak.flatten(hadron_flavour))
    flat_gentau_dr = np.asarray(ak.flatten(gentau_dr))

    category = np.full(len(flat_pt), "light", dtype=object)
    category[flat_flavour == 5] = "b"
    category[flat_gentau_dr < 0.4] = "tau"  # overrides "b" if both -- gen-tau match wins

    # correctionlib's vectorized evaluate() only allows array args for
    # int/real inputs, not the string category -- so evaluate per-category
    # group (mask), not with one big per-jet array of category strings.
    sf = np.empty(len(flat_pt))
    for cat in ("b", "light", "tau"):
        mask = category == cat
        if np.any(mask):
            sf[mask] = corr.evaluate(cat, flat_eta[mask], flat_pt[mask])

    corrected_pt = flat_pt * sf
    return ak.unflatten(corrected_pt, counts)


def apply_ak4_data_correction(jet_pt, jet_eta, btag_score, collection="plain"):
    """Example: dedicated per-category correction for AK4 jets in real data,
    using the offline b-tag score to pick the category (no truth available).
    collection: "plain" or "CHS"."""
    cset = correctionlib.CorrectionSet.from_file(CORRECTIONS_FILE)
    name = "AK4_plain_Data2024" if collection == "plain" else "AK4_CHS_Data2024"
    corr = cset[name]

    flat_pt = np.asarray(ak.flatten(jet_pt))
    counts = ak.num(jet_pt)
    flat_eta = np.asarray(ak.flatten(jet_eta))
    flat_score = np.asarray(ak.flatten(btag_score))

    tagged = flat_score >= 0.5
    sf = np.empty(len(flat_pt))
    if np.any(tagged):
        sf[tagged] = corr.evaluate("btag", flat_eta[tagged], flat_pt[tagged])
    if np.any(~tagged):
        sf[~tagged] = corr.evaluate("nobtag", flat_eta[~tagged], flat_pt[~tagged])

    corrected_pt = flat_pt * sf
    return ak.unflatten(corrected_pt, counts)


def apply_inclusive_correction(jet_pt, jet_eta, correction_name):
    """Simplest case: no per-jet category needed, just the flat inclusive
    map for a given collection/sample (correction_name is one of AK4_plain_MC,
    AK4_CHS_MC, AK8_MC, AK4_plain_Data2024, AK4_CHS_Data2024, AK8_Data2024,
    or the *_Data2025C_ref single-era reference equivalents)."""
    cset = correctionlib.CorrectionSet.from_file(CORRECTIONS_FILE)
    corr = cset[correction_name]

    flat_pt = np.asarray(ak.flatten(jet_pt))
    counts = ak.num(jet_pt)
    flat_eta = np.asarray(ak.flatten(jet_eta))

    sf = corr.evaluate("inclusive", flat_eta, flat_pt)
    corrected_pt = flat_pt * sf
    return ak.unflatten(corrected_pt, counts)


"""
RDataFrame (C++/PyROOT) usage
------------------------------
correctionlib ships a C++ header/library alongside the python package (same
CMSSW external as used by JSONPOG-derived JEC/b-tag SF tools). From PyROOT:

    import ROOT
    ROOT.gInterpreter.Declare('''
    #include "correction.h"
    auto cset = correction::CorrectionSet::from_file("results/scoutingPUPPI_corrections.json.gz");
    auto ak4_plain_data = cset->at("AK4_plain_Data2024");

    float apply_ak4_plain_data_sf(float pt, float eta, float btagScore) {
        std::string category = btagScore >= 0.5 ? "btag" : "nobtag";
        double sf = ak4_plain_data->evaluate({category, eta, pt});
        return pt * sf;
    }
    ''')

    df = ROOT.RDataFrame("Events", "scoutingnano_data.root")
    df = df.Define("Jet_correctedPt",
                    "ROOT::VecOps::RVec<float> out; "
                    "for (size_t i = 0; i < ScoutingPFJetRecluster2_pt.size(); ++i) "
                    "  out.push_back(apply_ak4_plain_data_sf("
                    "      ScoutingPFJetRecluster2_pt[i], ScoutingPFJetRecluster2_eta[i], "
                    "      Jet_btagUParTAK4B[i]));"  # or whatever the b-tag branch is called downstream
                    "return out;")

Note the C++ evaluate() call takes a correction::Variable::type-erased list
in the SAME ORDER as the Correction's `inputs` (systematic, JetEta, JetPt
here) -- same order as the python evaluate() calls above.
"""
