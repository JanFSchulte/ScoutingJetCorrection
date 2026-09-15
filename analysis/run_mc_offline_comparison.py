#!/usr/bin/env python3
"""CLI entry point: dR-match scouting jets to offline (non-scouting) jets
within the SAME MC MiniAOD-derived ScoutingNanoAOD file, using the OfflineJet/
OfflineFatJet tables added by customiseScoutingNanoWithOfflineJets() (see
PhysicsTools/PatFromScouting/scoutingToMiniAODDerivedCollections_cff.py and
its wiring into scoutingnano_mc_standalone2.py). Unlike run_data_matching.py
(the real-data path), there's no cross-file event-id join here -- scouting
and offline jets already share the same event/entry, so this only needs
data_matching.build_comparison_table_mc()'s per-event dR match.

Input files must have been produced by a cfg that calls
customiseScoutingNanoWithOfflineJets(); older/already-produced ScoutingNanoAOD
without that customisation has no OfflineJet branches and must be
regenerated (see that function's docstring), not reprocessed by this script.

Usage:
    python run_mc_offline_comparison.py \\
        --files scoutingnano_mc.root \\
        --output mc_offline_comparison.npz

    # AK8:
    python run_mc_offline_comparison.py \\
        --files scoutingnano_mc.root \\
        --scout-collection ScoutingFatPFJetRecluster2 \\
        --offline-prefix OfflineFatJet \\
        --output mc_offline_comparison_ak8.npz
"""

import argparse
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import data_matching


def _collect_files(patterns):
    files = []
    for p in patterns:
        matches = sorted(glob.glob(p))
        files.extend(matches if matches else [p])
    if not files:
        raise SystemExit("No input files found for patterns: %s" % patterns)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="+", required=True,
        help="ScoutingNanoAOD MC file(s)/glob(s), produced with "
             "customiseScoutingNanoWithOfflineJets() in the cfg.")
    parser.add_argument("--scout-collection", default="ScoutingPFJetRecluster2",
        help="Scouting jet table name= to compare (default: "
             "ScoutingPFJetRecluster2; see io.BASELINE_COLLECTIONS and "
             "io.BASELINE_COLLECTIONS_AK8 for other options).")
    parser.add_argument("--offline-prefix", default="OfflineJet",
        help="Offline jet table name= to compare against (default: "
             "OfflineJet; use OfflineFatJet for AK8).")
    parser.add_argument("--fields", nargs="+", default=None,
        help="Fields carried from BOTH sides into the output table (default: "
             "pt eta phi mass rawFactor -- present on both sides).")
    parser.add_argument("--off-fields", nargs="+", default=None,
        help="Additional offline-only fields with no scouting equivalent (e.g. "
             "hadronFlavour, partonFlavour -- truth flavour info only stored on "
             "the offline side) -- appended to --fields for the offline side only.")
    parser.add_argument("--max-dr", type=float, default=0.4,
        help="Max dR for jet-jet matching within an event (default 0.4).")
    parser.add_argument("--gentau-collection", default=None,
        help="If given (e.g. GenVisTau), adds an off_genVisTau_dr column: dR "
             "from each matched offline jet to the closest gen hadronic-tau in "
             "the same event (99.0 if none). Requires the input file(s) to have "
             "been produced with customiseScoutingNanoWithGenVisTau() in the cfg.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    args = parser.parse_args()

    files = _collect_files(args.files)
    print("Matching %d file(s): %s" % (len(files), files))

    fields = args.fields or list(data_matching._MC_COMMON_FIELDS)
    off_fields = fields + [f for f in (args.off_fields or []) if f not in fields]

    table = data_matching.build_comparison_table_mc(
        files, scout_collection=args.scout_collection,
        offline_prefix=args.offline_prefix, scout_fields=fields, off_fields=off_fields,
        max_dr=args.max_dr, gentau_collection=args.gentau_collection)
    print("  %d matched jet pairs" % len(table["dr"]))

    np.savez(args.output, **table)
    print("Wrote %s" % args.output)


if __name__ == "__main__":
    main()
