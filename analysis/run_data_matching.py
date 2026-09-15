#!/usr/bin/env python3
"""CLI entry point: match a scouting NanoAOD sample against existing central
offline NanoAOD (JetMET0/1 or Muon0/1) by (run, luminosityBlock, event), and
write the resulting dR-matched jet-pair comparison table.

No new production is involved -- this reads centrally-produced offline
NanoAOD directly over xrootd (see analysis/data_matching.py for why that's
sufficient and how the three-stage pipeline is scoped).

Usage:
    python run_data_matching.py \\
        --scouting-files scouting_nano_data.root \\
        --offline-datasets /JetMET0/Run2024D-MINIv6NANOv15-v1/NANOAOD \\
                           /JetMET1/Run2024D-MINIv6NANOv15-v1/NANOAOD \\
        --output matched_jets_2024D.npz
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
    parser.add_argument("--scouting-files", nargs="+", required=True,
        help="Scouting NanoAOD file(s)/glob(s) to match (local paths).")
    parser.add_argument("--offline-datasets", nargs="+", required=True,
        help="DAS dataset name(s) for the offline PD -- pass both siblings "
             "of a trigger-hash-split PD, e.g. /JetMET0/.../NANOAOD "
             "/JetMET1/.../NANOAOD, since it isn't confirmed up front which "
             "half a given event lands in.")
    parser.add_argument("--scout-collection", default="ScoutingPFJetRecluster2",
        help="Scouting jet table name= to compare (default: "
             "ScoutingPFJetRecluster2 (AK4); see io.BASELINE_COLLECTIONS for the "
             "CHS alternative, or use ScoutingFatPFJetRecluster for AK8, together "
             "with --offline-prefix FatJet --fields pt eta phi mass area).")
    parser.add_argument("--offline-prefix", default="Jet",
        help="Offline jet table name= to compare against (default: Jet (AK4); "
             "use FatJet for AK8 -- official NanoAOD's standard AK8 PUPPI jet name).")
    parser.add_argument("--fields", nargs="+", default=None,
        help="Fields pulled from BOTH sides (default: pt eta phi mass rawFactor, plus "
             "off-side-only area -- matches the historical AK4 default). Pass "
             "e.g. 'pt eta phi mass area' for AK8: the scouting AK8 collection has no "
             "rawFactor branch, unlike AK4.")
    parser.add_argument("--redirector", default=data_matching.DEFAULT_REDIRECTOR,
        help="xrootd redirector prefix for offline files (default: global "
             "redirector; use a site-local one, e.g. "
             "root://cms-xrootd.rcac.purdue.edu/, for speed).")
    parser.add_argument("--max-dr", type=float, default=0.4,
        help="Max dR for jet-jet matching within a matched event (default 0.4).")
    parser.add_argument("--max-workers", type=int, default=data_matching.DEFAULT_MAX_WORKERS,
        help="Concurrent threads for DAS queries and xrootd/local file opens in every "
             "stage (default %d). Each is network-latency-bound, not CPU-bound, so this "
             "gives close to linear speedup up to the point of saturating the redirector "
             "-- raise it if using a site-local redirector (see --redirector)."
             % data_matching.DEFAULT_MAX_WORKERS)
    parser.add_argument("--cache-dir", default=None,
        help="Directory to cache each offline file's (run,luminosityBlock,event) key scan "
             "(Stage 2, the dominant cost) by LFN. Unset by default (no caching); set this "
             "to make repeat runs over the same candidate files -- e.g. while tuning "
             "--max-dr or --scout-collection -- skip the xrootd scan entirely on a cache "
             "hit.")
    parser.add_argument("--output", required=True, help="Output .npz path.")
    args = parser.parse_args()

    scouting_files = _collect_files(args.scouting_files)

    print("Stage 1: reading event keys from %d scouting file(s)..." % len(scouting_files))
    scout_df = data_matching.scouting_event_keys(scouting_files, max_workers=args.max_workers)
    runs = sorted(int(r) for r in scout_df["run"].unique())
    print("  %d scouting events, runs: %s" % (len(scout_df), runs))

    print("Stage 2: discovering offline files for %d run(s) across %d dataset(s)..." %
          (len(runs), len(args.offline_datasets)))
    offline_files = data_matching.find_offline_files(args.offline_datasets, runs,
                                                       max_workers=args.max_workers)
    print("  %d candidate offline files" % len(offline_files))

    matches_df = data_matching.match_offline_files(
        scout_df, offline_files, redirector=args.redirector,
        max_workers=args.max_workers, cache_dir=args.cache_dir)
    print("  %d matched events" % len(matches_df))

    print("Stage 3: extracting jets and dR-matching matched events...")
    fields = args.fields  # None -> build_comparison_table's AK4 defaults
    table = data_matching.build_comparison_table(
        matches_df, scout_collection=args.scout_collection, offline_prefix=args.offline_prefix,
        scout_fields=fields, off_fields=fields,
        redirector=args.redirector, max_dr=args.max_dr, max_workers=args.max_workers)
    print("  %d matched jet pairs" % len(table["dr"]))

    np.savez(args.output, **table)
    print("Wrote %s" % args.output)


if __name__ == "__main__":
    main()
