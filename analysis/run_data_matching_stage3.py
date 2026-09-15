#!/usr/bin/env python3
"""Re-run just Stage 3 (jet extraction + dR matching) of the data-matching
pipeline (data_matching.build_comparison_table()) against an already-cached
Stage 1+2 event-provenance table (a matches.npz written by
run_data_matching_periods.py/submit_data_matching_slurm.py -- run,
luminosityBlock, event, scout_file, scout_entry, off_file, off_entry).

Stages 1-2 (scouting event keys, DAS offline-file discovery, run/lumi/event
join) are by far the most expensive part of the pipeline (one DAS query and
one xrootd key-scan per candidate offline file) and don't depend on which
jet collection/fields are pulled in Stage 3 -- so re-deriving a correction
for a DIFFERENT scouting jet collection (e.g. CHS instead of the plain
baseline) against the SAME already-matched events should skip straight to
Stage 3 instead of repeating Stages 1-2. This is exactly the ad hoc pattern
used to build the tagger-score comparison tables (2025C_tagger_ak4/ak8.npz);
formalized here as a reusable driver.

Usage:
    python run_data_matching_stage3.py \\
        --matches results/data_matching_periods/2025C/matches.npz \\
        --scout-collection ScoutingPFJetReclusterCHS \\
        --off-fields pt eta phi mass rawFactor area btagUParTAK4B \\
        --output results/data_matching_periods/2025C_ak4chs.npz
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import data_matching


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--matches", required=True,
        help="Cached Stage 1+2 matches.npz (run, luminosityBlock, event, "
             "scout_file, scout_entry, off_file, off_entry).")
    parser.add_argument("--scout-collection", default="ScoutingPFJetRecluster2")
    parser.add_argument("--offline-prefix", default="Jet")
    parser.add_argument("--fields", nargs="+", default=None,
        help="Fields pulled from BOTH sides (default: data_matching's AK4 "
             "defaults: pt eta phi mass rawFactor / +area offline-only).")
    parser.add_argument("--off-fields", nargs="+", default=None,
        help="Full explicit offline-side field list (overrides --fields for "
             "the offline side only, e.g. to add a tagger score column like "
             "btagUParTAK4B). scout-side still uses --fields.")
    parser.add_argument("--redirector", default=data_matching.DEFAULT_REDIRECTOR)
    parser.add_argument("--max-dr", type=float, default=0.4)
    parser.add_argument("--max-workers", type=int, default=data_matching.DEFAULT_MAX_WORKERS)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("Loading cached matches from %s" % args.matches)
    d = np.load(args.matches, allow_pickle=True)
    matches_df = pd.DataFrame({k: d[k] for k in d.files})
    print("  %d matched events" % len(matches_df))

    scout_fields = args.fields
    off_fields = args.off_fields or args.fields

    table = data_matching.build_comparison_table(
        matches_df, scout_collection=args.scout_collection, offline_prefix=args.offline_prefix,
        scout_fields=scout_fields, off_fields=off_fields,
        redirector=args.redirector, max_dr=args.max_dr, max_workers=args.max_workers)
    print("  %d matched jet pairs" % len(table["dr"]))

    np.savez(args.output, **table)
    print("Wrote %s" % args.output)


if __name__ == "__main__":
    main()
