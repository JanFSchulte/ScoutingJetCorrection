#!/usr/bin/env python3
"""Package all derived scouting-vs-offline AK4/AK8 response corrections
(jet_correction.derive_response_curve() maps, per collection/sample/category)
into a single correctionlib CorrectionSet -- see correctionlib_export.py for
how a curve becomes a correctionlib node.

One Correction per (jet collection, sample); each takes a string "systematic"
category ("inclusive" plus whatever dedicated categories exist for that
group, e.g. "b"/"light" for MC truth flavour or "btag"/"nobtag" for
data-driven tagger-score splits) plus (JetEta, JetPt) of the RAW scouting
jet, and returns a multiplicative SF: corrected_pt = SF * raw_pt.

MC groups' "inclusive" curve is derived here on the fly (never saved
separately by run_flavor_correction.py) from the same comparison table the
dedicated per-category curves came from, via jet_correction.
derive_response_curve() with its default binning/min_stat -- identical to
how plot_ak4_b_tau_correction.py/plot_data_btag_correction.py compute the
"inclusive-corrected" curve they compare against.

Provenance note: the AK4 and AK4CHS truth-flavour/gentau curves already on
disk were verified (bin-count cross-check against jet_correction.
derive_response_curve_by_category on the source table) to come from the
v7 (GenVisTau-enabled) HH->bbtautau tables with default binning. The AK8
truth-flavour/gentau curves did NOT reproduce that way -- they were derived
against an older/differently-binned setup -- so this script's GROUPS list
points at freshly-regenerated results/flavor_correction_hhbbtautau_v7_ak8/
and results/gentau_correction_hhbbtautau_v7_ak8/ (same run_flavor_correction.py
invocation and default binning as the AK4/AK4CHS ones) instead of the stale
results/flavor_correction_hhbbtautau_ak8/ and results/gentau_correction_hhbbtautau_ak8/.

Usage:
    python build_correctionlib_set.py --output results/scoutingPUPPI_corrections.json.gz
"""

import argparse
import gzip
import os
import sys

import numpy as np
import correctionlib.schemav2 as cs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import jet_correction
import correctionlib_export as clexp


GROUPS = [
    {
        "name": "AK4_plain_MC",
        "description": ("HH->bbtautau MC (v7, GenVisTau-enabled): ScoutingPFJetRecluster2 "
                         "(plain PUPPI-reclustered AK4) vs offline OfflineJet response "
                         "correction. Categories: inclusive, b/light (truth off_hadronFlavour), "
                         "tau/nontau (gen-matched hadronic tau, dR(offline jet, GenVisTau) < 0.4)."),
        "table": "results/mc_offline_comparison_hhbbtautau_v7_ak4.npz",
        "categories": {
            "b": "results/flavor_correction_hhbbtautau_v7_ak4/response_curve_b.npz",
            "light": "results/flavor_correction_hhbbtautau_v7_ak4/response_curve_light.npz",
            "tau": "results/gentau_correction_hhbbtautau_ak4/response_curve_tagged.npz",
            "nontau": "results/gentau_correction_hhbbtautau_ak4/response_curve_untagged.npz",
        },
    },
    {
        "name": "AK4_CHS_MC",
        "description": ("HH->bbtautau MC (v7, GenVisTau-enabled): ScoutingPFJetReclusterCHS "
                         "(CHS AK4) vs offline OfflineJet response correction. Categories: "
                         "inclusive, b/light (truth off_hadronFlavour), tau/nontau "
                         "(gen-matched hadronic tau, dR(offline jet, GenVisTau) < 0.4)."),
        "table": "results/mc_offline_comparison_hhbbtautau_v7_ak4chs.npz",
        "categories": {
            "b": "results/flavor_correction_hhbbtautau_v7_ak4chs/response_curve_b.npz",
            "light": "results/flavor_correction_hhbbtautau_v7_ak4chs/response_curve_light.npz",
            "tau": "results/gentau_correction_hhbbtautau_ak4chs/response_curve_tagged.npz",
            "nontau": "results/gentau_correction_hhbbtautau_ak4chs/response_curve_untagged.npz",
        },
    },
    {
        "name": "AK8_MC",
        "description": ("HH->bbtautau MC (v7, GenVisTau-enabled): ScoutingFatPFJetRecluster "
                         "(AK8) vs offline OfflineFatJet response correction. Categories: "
                         "inclusive, b/light (truth off_hadronFlavour), tau/nontau "
                         "(gen-matched hadronic tau, dR(offline jet, GenVisTau) < 0.4). Note: "
                         "AK8 response has a much larger low-pt (< ~150 GeV) bias than AK4 -- "
                         "a real substructure-reconstruction effect, not a packaging artifact."),
        "table": "results/mc_offline_comparison_hhbbtautau_v7_ak8.npz",
        "categories": {
            "b": "results/flavor_correction_hhbbtautau_v7_ak8/response_curve_b.npz",
            "light": "results/flavor_correction_hhbbtautau_v7_ak8/response_curve_light.npz",
            "tau": "results/gentau_correction_hhbbtautau_v7_ak8/response_curve_tagged.npz",
            "nontau": "results/gentau_correction_hhbbtautau_v7_ak8/response_curve_untagged.npz",
        },
    },
    {
        "name": "AK4_plain_Data2024",
        "description": ("Real 2024G+H+I JetMET/Muon data (the run period used for analysis "
                         "as of this derivation -- see AK4_plain_Data2025C_ref for the 2025C "
                         "single-era reference point instead): ScoutingPFJetRecluster2 (plain "
                         "PUPPI-reclustered AK4) vs central offline NanoAOD Jet response "
                         "correction, matched by (run, luminosityBlock, event). No truth "
                         "flavour in data -- categories are inclusive, btag/nobtag (offline "
                         "UParT b-tag score off_btagUParTAK4B >= 0.5). 2024G/H/I were checked "
                         "for scale/resolution stability against each other (see "
                         "plot_era_stability.py) before combining -- all three are mutually "
                         "consistent; 2025C is a separate, systematically shifted period and "
                         "is NOT combined in here."),
        "table": "results/data_matching_periods/2024_tagger_ak4.npz",
        "categories": {
            "btag": "results/tagger_correction_2024_ak4_btag/response_curve_tagged.npz",
            "nobtag": "results/tagger_correction_2024_ak4_btag/response_curve_untagged.npz",
        },
    },
    {
        "name": "AK4_CHS_Data2024",
        "description": ("Real 2024G+H+I JetMET/Muon data: ScoutingPFJetReclusterCHS (CHS "
                         "AK4) vs central offline NanoAOD Jet response correction, matched "
                         "by (run, luminosityBlock, event). Categories: inclusive, btag/"
                         "nobtag (offline UParT b-tag score off_btagUParTAK4B >= 0.5)."),
        "table": "results/data_matching_periods/2024_ak4chs_tagger.npz",
        "categories": {
            "btag": "results/tagger_correction_2024_ak4chs_btag/response_curve_tagged.npz",
            "nobtag": "results/tagger_correction_2024_ak4chs_btag/response_curve_untagged.npz",
        },
    },
    {
        "name": "AK8_Data2024",
        "description": ("Real 2024G+H+I JetMET/Muon data: ScoutingFatPFJetRecluster (AK8) "
                         "vs central offline NanoAOD FatJet response correction, matched by "
                         "(run, luminosityBlock, event). Categories: inclusive, btag/nobtag "
                         "(offline ParticleNet score off_particleNet_XbbVsQCD >= 0.5 -- 2024 "
                         "offline NanoAOD predates the GlobalParT tagger used for the 2025C "
                         "AK8_Data2025C_ref group's off_globalParT3_Xbb; ParticleNet's "
                         "XbbVsQCD is the same-purpose Xbb-vs-QCD discriminant available for "
                         "this era)."),
        "table": "results/data_matching_periods/2024_tagger_ak8.npz",
        "categories": {
            "btag": "results/tagger_correction_2024_ak8_btag/response_curve_tagged.npz",
            "nobtag": "results/tagger_correction_2024_ak8_btag/response_curve_untagged.npz",
        },
    },
    {
        "name": "AK4_plain_Data2025C_ref",
        "description": ("Real 2025C JetMET/Muon data (kept as a single-era reference point "
                         "alongside AK4_plain_Data2024, the current analysis's primary data "
                         "correction -- 2025C is NOT combined with 2024 since it showed a "
                         "systematic scale/resolution offset from 2024G/H/I, see "
                         "plot_era_stability.py): ScoutingPFJetRecluster2 (plain PUPPI-"
                         "reclustered AK4) vs central offline NanoAOD Jet response "
                         "correction, matched by (run, luminosityBlock, event). Categories: "
                         "inclusive, btag/nobtag (offline UParT b-tag score "
                         "off_btagUParTAK4B >= 0.5)."),
        "table": "results/data_matching_periods/2025C_tagger_ak4.npz",
        "categories": {
            "btag": "results/tagger_correction_2025C_ak4_btag/response_curve_tagged.npz",
            "nobtag": "results/tagger_correction_2025C_ak4_btag/response_curve_untagged.npz",
        },
    },
    {
        "name": "AK4_CHS_Data2025C_ref",
        "description": ("Real 2025C JetMET/Muon data (single-era reference, see "
                         "AK4_plain_Data2025C_ref): ScoutingPFJetReclusterCHS (CHS AK4) "
                         "vs central offline NanoAOD Jet response correction, matched by "
                         "(run, luminosityBlock, event). Categories: inclusive, btag/nobtag "
                         "(offline UParT b-tag score off_btagUParTAK4B >= 0.5)."),
        "table": "results/data_matching_periods/2025C_ak4chs_tagger.npz",
        "categories": {
            "btag": "results/tagger_correction_2025C_ak4chs_btag/response_curve_tagged.npz",
            "nobtag": "results/tagger_correction_2025C_ak4chs_btag/response_curve_untagged.npz",
        },
    },
    {
        "name": "AK8_Data2025C_ref",
        "description": ("Real 2025C JetMET/Muon data (single-era reference, see "
                         "AK4_plain_Data2025C_ref): ScoutingFatPFJetRecluster (AK8) vs "
                         "central offline NanoAOD FatJet response correction, matched by "
                         "(run, luminosityBlock, event). Categories: inclusive, btag/nobtag "
                         "(offline ParT score off_globalParT3_Xbb >= 0.5)."),
        "table": "results/data_matching_periods/2025C_tagger_ak8.npz",
        "categories": {
            "btag": "results/tagger_correction_2025C_ak8_btag/response_curve_tagged.npz",
            "nobtag": "results/tagger_correction_2025C_ak8_btag/response_curve_untagged.npz",
        },
    },
]


def build_group_correction(group, n_pt_eval):
    d = np.load(group["table"])
    table = {k: d[k] for k in d.files}
    inclusive_curve = jet_correction.derive_response_curve(table)

    curves_by_category = {"inclusive": inclusive_curve}
    for cat_name, path in group["categories"].items():
        curves_by_category[cat_name] = jet_correction.load_response_curve(path)

    return clexp.build_correction(group["name"], group["description"],
                                   curves_by_category, n_pt_eval=n_pt_eval)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", default="results/scoutingPUPPI_corrections.json.gz")
    parser.add_argument("--n-pt-eval", type=int, default=60,
        help="Number of log-spaced raw-pt bins baked into each correctionlib "
             "MultiBinning (default 60 -- finer than the ~12 bins the response "
             "curves themselves use, see correctionlib_export.py docstring).")
    args = parser.parse_args()

    corrections = []
    for group in GROUPS:
        print("Building %s ..." % group["name"])
        corrections.append(build_group_correction(group, args.n_pt_eval))

    cset = cs.CorrectionSet(
        schema_version=2,
        description=("Scouting PUPPI/CHS AK4 and AK8 jet pt corrections, derived from "
                      "dR-matched scouting-vs-offline jets (HH->bbtautau MC for truth-flavour "
                      "and gen-tau categories, real 2025C data for tagger-score categories). "
                      "See ScoutingPuppiCalibration/Calibration/analysis/jet_correction.py "
                      "for the derivation method."),
        corrections=corrections,
    )

    payload = cset.model_dump_json(exclude_unset=True, indent=2)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    if args.output.endswith(".gz"):
        with gzip.open(args.output, "wt") as f:
            f.write(payload)
    else:
        with open(args.output, "w") as f:
            f.write(payload)
    print("Wrote %d corrections to %s" % (len(corrections), args.output))


if __name__ == "__main__":
    main()
