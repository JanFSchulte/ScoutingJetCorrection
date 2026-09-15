#!/usr/bin/env python3
"""Run submit_data_matching_slurm.py once per data-taking period, for a
scouting-vs-offline correction stability study across time.

Periods and their scouting/offline dataset names come from
data_matching_periods.py -- see that module's docstring for which periods
are ready (scouting production already exists) vs. missing it entirely.

Each period gets its own work-dir/output under --base-dir, so results don't
collide. Runs periods SEQUENTIALLY (not in parallel): each period's Stage 1-2
already does its own multi-threaded DAS/xrootd scan, and running several at
once would multiply load on the same shared redirector for no benefit here.

Usage:
    # sanity-check all ready periods (Stage 1-2 + staging + manifest only,
    # no Slurm submission) before committing to a real run:
    python run_data_matching_periods.py --dry-run

    # actually submit the Slurm arrays, one period at a time:
    python run_data_matching_periods.py --periods 2024G 2024H 2024I 2025C
"""

import argparse
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import data_matching_periods as periods_mod

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SUBMIT_SCRIPT = os.path.join(THIS_DIR, "submit_data_matching_slurm.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--periods", nargs="+", default=None,
        help="Period names to run (default: every key in READY_PERIODS, i.e. "
             "%s)." % ", ".join(periods_mod.READY_PERIODS))
    parser.add_argument("--base-dir", default="results/data_matching_periods",
        help="Parent directory for per-period work-dir/output (default: "
             "results/data_matching_periods).")
    parser.add_argument("--max-workers", type=int, default=16)
    parser.add_argument("--max-scouting-files", type=int, default=60,
        help="Cap staged/matched scouting files per period, evenly sampled across "
             "that period's full file list (default 60, matching the 2025C reference "
             "point's 56 files, so every period in the stability comparison has "
             "roughly comparable statistics rather than being dominated by whichever "
             "era happens to have the most files -- some 2024 eras have 9,000+ files "
             "per part). Run submit_data_matching_slurm.py directly (which stages "
             "every file when --max-scouting-files is omitted) for full statistics "
             "on one period instead of this multi-period driver.")
    parser.add_argument("--mem", default="4G",
        help="Default 4G was too low for the 2024 eras' Stage 3 tasks (OOM'd on "
             "25-57%% of chunks the first time this ran) -- pass a higher value, "
             "e.g. 8G-12G, especially when combined with --resume.")
    parser.add_argument("--time", default="30")
    parser.add_argument("--throttle", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true",
        help="Passed through to submit_data_matching_slurm.py for every period -- "
             "Stage 1-2 + staging + manifest only, no Slurm submission.")
    parser.add_argument("--resume", action="store_true",
        help="Passed through to submit_data_matching_slurm.py for every period -- reuse "
             "each period's existing matches.npz/manifest_full.txt and only resubmit "
             "chunks still missing from chunks/ (e.g. after OOM/timeout failures).")
    args = parser.parse_args()

    period_names = args.periods or list(periods_mod.READY_PERIODS.keys())
    unknown = [p for p in period_names if p not in periods_mod.READY_PERIODS]
    if unknown:
        raise SystemExit("Unknown/not-ready period(s) %s -- ready periods: %s. "
                          "See data_matching_periods.MISSING_SCOUTING_PRODUCTION "
                          "for periods needing new CRAB scouting production first."
                          % (unknown, list(periods_mod.READY_PERIODS)))

    print("Running %d period(s): %s" % (len(period_names), period_names), flush=True)

    results = {}
    for name in period_names:
        cfg = periods_mod.READY_PERIODS[name]
        work_dir = os.path.join(args.base_dir, name)
        output = os.path.join(args.base_dir, "%s.npz" % name)

        cmd = [
            sys.executable, SUBMIT_SCRIPT,
            "--scouting-dataset"] + cfg["scouting_datasets"] + [
            "--offline-datasets"] + cfg["offline_datasets"] + [
            "--work-dir", work_dir,
            "--output", output,
            "--max-workers", str(args.max_workers),
            "--max-scouting-files", str(args.max_scouting_files),
            "--mem", args.mem,
            "--time", args.time,
            "--throttle", str(args.throttle),
        ]
        if args.dry_run:
            cmd.append("--dry-run")
        if args.resume:
            cmd.append("--resume")

        print("\n=== %s (%d scouting dataset(s), runs %s) ==="
              % (name, len(cfg["scouting_datasets"]), cfg["run_range"]), flush=True)
        print("+ %s" % " ".join(cmd), flush=True)
        ret = subprocess.run(cmd)
        results[name] = ret.returncode
        if ret.returncode != 0:
            print("!! %s exited with code %d -- continuing with remaining periods"
                  % (name, ret.returncode), flush=True)

    print("\n=== summary ===", flush=True)
    for name, code in results.items():
        print("  %-8s %s" % (name, "ok" if code == 0 else "FAILED (exit %d)" % code), flush=True)

    if any(c != 0 for c in results.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
