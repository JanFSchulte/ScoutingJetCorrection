#!/usr/bin/env python3
"""Slurm-array chunk worker for Stage 3 of the data-matching pipeline.

Not meant to be run by hand for a full dataset -- see submit_data_matching_
slurm.py, which runs Stages 1-2 once (cheap: event-key extraction + offline
file discovery/join), then submits one array task per scouting file via
this script for Stage 3 (targeted jet extraction + dR matching, the part
that scales with the full dataset's matched-event count).

Standalone usage (e.g. to reprocess one file by hand):
    python data_matching_stage3_chunk.py \\
        --matches matches.npz --scout-file /depot/.../scouting_nano_data_1.root \\
        --outfile chunk_1.npz
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import data_matching


def load_matches(path):
    """Inverse of submit_data_matching_slurm.py's save_matches() -- see that
    module for why this is a plain npz (of parallel arrays) rather than a
    parquet/CSV file (no pyarrow in this environment; npz keeps everything
    in the same format the rest of this pipeline already uses)."""
    d = np.load(path, allow_pickle=False)
    return pd.DataFrame({
        "run": d["run"], "luminosityBlock": d["luminosityBlock"], "event": d["event"],
        "scout_file": d["scout_file"], "scout_entry": d["scout_entry"],
        "off_file": d["off_file"], "off_entry": d["off_entry"],
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--matches", required=True,
        help="matches_df .npz written by submit_data_matching_slurm.py's Stage 1-2 pass")
    ap.add_argument("--scout-file", required=True,
        help="Only process rows whose scout_file equals this path exactly -- must be the "
             "same (staged, compute-node-visible) path used when matches_df was built, "
             "not a re-derived one.")
    ap.add_argument("--scout-collection", default="ScoutingPFJetRecluster2",
        help="Default: AK4. Use ScoutingFatPFJetRecluster for AK8, together with "
             "--offline-prefix FatJet --fields pt eta phi mass area.")
    ap.add_argument("--offline-prefix", default="Jet", help="Default: Jet (AK4); use FatJet for AK8.")
    ap.add_argument("--fields", nargs="+", default=None,
        help="Fields pulled from both sides by default (default: build_comparison_table's "
             "AK4 defaults). Scout side always uses this list.")
    ap.add_argument("--off-fields", nargs="+", default=None,
        help="Offline-side field list, overriding --fields for that side only -- e.g. to "
             "add a tagger score column (btagUParTAK4B, particleNet_XbbVsQCD) that has no "
             "scouting-side equivalent. Defaults to --fields if omitted.")
    ap.add_argument("--redirector", default=data_matching.DEFAULT_REDIRECTOR)
    ap.add_argument("--max-dr", type=float, default=0.4)
    ap.add_argument("--max-workers", type=int, default=data_matching.DEFAULT_MAX_WORKERS)
    ap.add_argument("--outfile", required=True)
    args = ap.parse_args()

    matches_df = load_matches(args.matches)
    sel = matches_df[matches_df["scout_file"] == args.scout_file]
    print("%d matched row(s) for %s" % (len(sel), args.scout_file), flush=True)

    table = data_matching.build_comparison_table(
        sel, scout_collection=args.scout_collection, offline_prefix=args.offline_prefix,
        scout_fields=args.fields, off_fields=(args.off_fields or args.fields),
        redirector=args.redirector, max_dr=args.max_dr, max_workers=args.max_workers)

    os.makedirs(os.path.dirname(os.path.abspath(args.outfile)), exist_ok=True)
    np.savez(args.outfile, **table)
    print("wrote %s (%d matched jet pairs)" % (args.outfile, len(table["dr"])), flush=True)


if __name__ == "__main__":
    main()
