#!/usr/bin/env python3
"""Run the data_matching.py pipeline at dataset scale via Slurm.

Stages 1-2 (scouting event-key extraction, offline file discovery, and the
run/lumi/event join -- analysis/data_matching.py's scouting_event_keys(),
find_offline_files(), match_offline_files()) run once, locally, right here:
they're already internally threaded (see data_matching.DEFAULT_MAX_WORKERS)
and, for the run ranges this pipeline deals with, cheap enough (a few
minutes, not hours) that Slurm sharding would only add overhead. Only Stage
3 (build_comparison_table -- targeted jet extraction + dR matching for the
matched events found above, the part that scales with the FULL scouting
dataset's size) is sharded across a Slurm array, one task per scouting file.

Two gotchas this works around (see memory/scouting_puppi_recalibration.md
and results/full_stats_work/ for how these were found the hard way on this
same cluster):
  1. /eos/purdue is NOT mounted on hammer-nodes compute nodes. Scouting
     files are staged (cp, or xrdcp as a fallback) from /eos/purdue to a
     /depot work directory before the array is submitted, and Stage 1's
     event keys are extracted from those staged paths -- so matches_df's
     scout_file column already holds compute-node-visible paths, with no
     path remapping needed downstream.
  2. The default grid proxy ($X509_USER_PROXY, typically /tmp/x509up_u<uid>)
     lives on the login node's node-local /tmp, invisible to compute nodes.
     It's copied to the (NFS-shared) work directory before submission, and
     data_matching_slurm_array.sh points X509_USER_PROXY there.

Usage:
    python submit_data_matching_slurm.py \\
        --scouting-dataset /ScoutingPFRun3/jschulte-ScoutingNano_Data_2025C_part1_v6-00000000000000000000000000000000/USER \\
        --dbs-instance prod/phys03 \\
        --offline-datasets /JetMET0/Run2025C-PromptReco-v1/NANOAOD /JetMET1/Run2025C-PromptReco-v1/NANOAOD \\
                           /Muon0/Run2025C-PromptReco-v1/NANOAOD /Muon1/Run2025C-PromptReco-v1/NANOAOD \\
        --work-dir results/data_matching_2025C_part1 \\
        --output results/data_matching_2025C_part1.npz

Pass --dry-run to run Stages 1-2, stage inputs, and write the manifest
without actually calling sbatch -- useful to sanity-check matched-event
counts and task count before spending cluster time.
"""

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analysis import data_matching

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ARRAY_SCRIPT = os.path.join(THIS_DIR, "data_matching_slurm_array.sh")

# Same account/partition/known-bad-node exclusion as every other Slurm job
# in this repo (results/full_stats_work/*.py) -- see
# memory/scouting_puppi_recalibration.md for how the bad nodes were found
# (hammer-f001/f006/f008 silently hang tasks with zero output).
DEFAULT_STAGE_WORKERS = 16
DEFAULT_ACCOUNT = "cms-express"
DEFAULT_PARTITION = "hammer-nodes"
DEFAULT_EXCLUDE = "hammer-f001,hammer-f006,hammer-f008"


def das_query(query):
    cmd = ["dasgoclient", "-query", query]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    return [line for line in out.splitlines() if line.strip()]


def _evenly_sample(items, max_n):
    """Deterministic subsample of max_n items spanning the full list (not
    just the first max_n) -- so a per-era file cap still covers the whole
    era's run range/luminosity rather than being biased toward whichever
    "part" dataset happens to sort first."""
    if max_n is None or len(items) <= max_n:
        return items
    step = len(items) / max_n
    idx = sorted(set(int(i * step) for i in range(max_n)))
    return [items[i] for i in idx]


def stage_scouting_files(datasets, instance, work_dir, redirector, max_files=None,
                          max_workers=DEFAULT_STAGE_WORKERS):
    """Resolve every scouting dataset's LFNs via DAS (datasets: one or more
    DBS dataset names -- e.g. all "part" datasets of one era, so a
    multi-part era can be matched in a single run), then copy each to
    work_dir/inputs/ in a thread pool -- preferring a direct /eos/purdue cp
    (fast, this is how ScoutingNanoProduction/submit_scoutingNano.py's CRAB
    output lands for this group's T2_US_Purdue jobs) and falling back to
    xrdcp over the given redirector for any file not locally present.

    max_files caps the number of files staged, evenly sampled across the
    full (possibly multi-dataset) LFN list -- some eras have thousands of
    files per part (e.g. 2024H: 13,010 across 2 parts), far more than needed
    for a stability-vs-time check; the reference 2025C point used 56 files,
    so a comparable cap keeps every period's statistics roughly the same
    order of magnitude instead of wildly uneven per-period sample sizes.

    Returns the list of staged local paths (compute-node-visible, under
    work_dir).
    """
    lfns = []
    for dataset in datasets:
        found = sorted(das_query("file dataset=%s instance=%s" % (dataset, instance)))
        if not found:
            raise SystemExit("No files found for dataset=%s instance=%s" % (dataset, instance))
        lfns.extend(found)
    lfns = _evenly_sample(lfns, max_files)

    inputs_dir = os.path.join(work_dir, "inputs")
    os.makedirs(inputs_dir, exist_ok=True)

    def _stage_one(lfn):
        dest = os.path.join(inputs_dir, os.path.basename(lfn))
        if os.path.exists(dest):
            return dest
        local_src = "/eos/purdue" + lfn
        if os.path.exists(local_src):
            shutil.copy(local_src, dest)
        else:
            url = redirector + lfn
            subprocess.run(["xrdcp", "-s", url, dest], check=True)
        return dest

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        staged = list(ex.map(_stage_one, lfns))
    return staged


def save_matches(path, matches_df):
    """npz, not parquet/CSV -- no pyarrow in this environment, and this
    keeps the hand-off artifact in the same format as everything else in
    this pipeline. String columns as fixed-width unicode arrays (not
    object/pickled) so data_matching_stage3_chunk.py can load with
    allow_pickle=False."""
    np.savez(
        path,
        run=matches_df["run"].to_numpy(),
        luminosityBlock=matches_df["luminosityBlock"].to_numpy(),
        event=matches_df["event"].to_numpy(),
        scout_file=np.array(matches_df["scout_file"].tolist()),
        scout_entry=matches_df["scout_entry"].to_numpy(),
        off_file=np.array(matches_df["off_file"].tolist()),
        off_entry=matches_df["off_entry"].to_numpy(),
    )


def stage_proxy(work_dir):
    src = os.environ.get("X509_USER_PROXY") or ("/tmp/x509up_u%d" % os.getuid())
    if not os.path.exists(src):
        raise SystemExit("No grid proxy found at %s -- run voms-proxy-init first." % src)
    proxy_dir = os.path.join(work_dir, "proxy")
    os.makedirs(proxy_dir, exist_ok=True)
    dest = os.path.join(proxy_dir, "x509up")
    shutil.copy(src, dest)
    os.chmod(dest, 0o600)
    return dest


def poll_job(job_id, n_tasks, poll_interval=20):
    t0 = time.time()
    while True:
        rq = subprocess.run(["squeue", "-j", job_id, "-h"], capture_output=True, text=True)
        if not rq.stdout.strip():
            break
        n_left = len(rq.stdout.strip().splitlines())
        print("[%.0f s] %d/%d task(s) still queued/running" % (time.time() - t0, n_left, n_tasks), flush=True)
        time.sleep(poll_interval)
    print("Array %s finished after %.1f min" % (job_id, (time.time() - t0) / 60.0), flush=True)

    racct = subprocess.run(["sacct", "-j", job_id, "--format=State", "-n", "-X"],
                            capture_output=True, text=True)
    states = [s.strip() for s in racct.stdout.strip().splitlines() if s.strip()]
    print("Task state breakdown: %s" % dict(Counter(states)), flush=True)


def merge_chunks_from_dir(chunks_dir, manifest_lines, output_path):
    """Concatenate every chunk .npz (exact, not an average of per-chunk
    stats -- same rationale as results/full_stats_merge.py). Reports any
    manifest entries with no corresponding chunk on disk: per
    memory/scouting_puppi_recalibration.md, trust what's actually on disk
    over squeue/sacct bookkeeping for the true missing-task set. Takes
    chunks_dir directly (not work_dir/"chunks") so a caller sharding a
    non-default Stage-3 variant into its own subdirectory -- see
    run_stage3_slurm.py -- can reuse this instead of duplicating it."""
    missing = []
    arrays_by_key = {}
    total_pairs = 0
    for scout_file in manifest_lines:
        base = os.path.splitext(os.path.basename(scout_file))[0]
        chunk_path = os.path.join(chunks_dir, base + ".npz")
        if not os.path.exists(chunk_path):
            missing.append(scout_file)
            continue
        d = np.load(chunk_path)
        for k in d.files:
            arrays_by_key.setdefault(k, []).append(d[k])
        total_pairs += len(d["dr"]) if "dr" in d.files else 0

    if missing:
        print("WARNING: %d/%d chunk(s) missing from disk (task failed or still running?):"
              % (len(missing), len(manifest_lines)))
        for m in missing:
            print("  %s" % m)

    merged = {k: np.concatenate(v) for k, v in arrays_by_key.items()}
    np.savez(output_path, **merged)
    print("Merged %d chunk(s) -> %s (%d matched jet pairs)"
          % (len(manifest_lines) - len(missing), output_path, total_pairs))
    return missing


def merge_chunks(work_dir, manifest_lines, output_path):
    """Default-variant wrapper around merge_chunks_from_dir (chunks_dir =
    work_dir/"chunks") -- this module's own main() below always uses that
    default location."""
    return merge_chunks_from_dir(os.path.join(work_dir, "chunks"), manifest_lines, output_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scouting-dataset", nargs="+", required=True,
        help="DBS dataset name(s) for the scouting sample -- one or more (e.g. every "
             "'part' dataset of one era, to match a full era in one run), e.g. "
             "/ScoutingPFRun3/jschulte-ScoutingNano_Data_2025C_part1_v6-.../USER")
    parser.add_argument("--dbs-instance", default="prod/phys03",
        help="DBS instance for --scouting-dataset (default prod/phys03, for "
             "CRAB/user-produced datasets; central datasets use prod/global).")
    parser.add_argument("--offline-datasets", nargs="+", required=True,
        help="DAS dataset name(s) for the offline PD -- pass all siblings of a "
             "trigger-hash-split PD (JetMET0+JetMET1, Muon0+Muon1) unless you've "
             "confirmed the scouting sample only used one DST_PFScouting_* path.")
    parser.add_argument("--scout-collection", default="ScoutingPFJetRecluster2")
    parser.add_argument("--redirector", default=data_matching.DEFAULT_REDIRECTOR,
        help="xrootd redirector for offline files and any xrdcp staging fallback "
             "(default global; use a site-local one for speed, e.g. "
             "root://cms-xrootd.rcac.purdue.edu/).")
    parser.add_argument("--max-dr", type=float, default=0.4)
    parser.add_argument("--max-workers", type=int, default=data_matching.DEFAULT_MAX_WORKERS,
        help="Threads for the local Stage 1-2 pass (default %d)." % data_matching.DEFAULT_MAX_WORKERS)
    parser.add_argument("--max-scouting-files", type=int, default=None,
        help="Cap the number of scouting files staged/matched, evenly sampled across "
             "the full (possibly multi-dataset) file list -- some eras have thousands "
             "of files; unset (default) stages every file, which for those eras will "
             "be slow and produce far more statistics than needed for a stability "
             "check. Compare to the 2025C reference point, which used 56 files.")
    parser.add_argument("--work-dir", required=True,
        help="Directory for staged inputs, matches.npz, manifest.txt, chunks/, "
             "logs/, proxy/ -- must be on /depot (NFS), not /tmp (node-local, "
             "invisible to compute nodes).")
    parser.add_argument("--output", required=True, help="Final merged .npz path.")
    parser.add_argument("--account", default=DEFAULT_ACCOUNT)
    parser.add_argument("--partition", default=DEFAULT_PARTITION)
    parser.add_argument("--exclude", default=DEFAULT_EXCLUDE,
        help="Comma-separated nodes to exclude (default: known-bad nodes on "
             "hammer-nodes, see memory/scouting_puppi_recalibration.md).")
    parser.add_argument("--mem", default="4G")
    parser.add_argument("--time", default="30", help="Per-task time limit in minutes (default 30).")
    parser.add_argument("--throttle", type=int, default=100,
        help="Max concurrent array tasks (SLURM %%N syntax, default 100).")
    parser.add_argument("--dry-run", action="store_true",
        help="Do Stages 1-2, stage inputs, and write the manifest, but don't submit "
             "or poll the Slurm array.")
    parser.add_argument("--resume", action="store_true",
        help="Skip staging + Stage 1-2 and reuse the existing matches.npz/manifest_full.txt "
             "in --work-dir (from a previous run of this same command); submit the Slurm "
             "array only for scout_files whose chunk is still missing from chunks/ -- e.g. "
             "after some tasks OOM'd or timed out. Raise --mem/--time for the retry.")
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    if work_dir.startswith("/tmp"):
        raise SystemExit("--work-dir must not be under /tmp (node-local, invisible to "
                          "compute nodes) -- use a path under /depot.")
    os.makedirs(work_dir, exist_ok=True)

    matches_path = os.path.join(work_dir, "matches.npz")
    manifest_full_path = os.path.join(work_dir, "manifest_full.txt")
    manifest_path = os.path.join(work_dir, "manifest.txt")

    if args.resume:
        if not (os.path.exists(matches_path) and os.path.exists(manifest_full_path)):
            raise SystemExit("--resume requires an existing matches.npz and manifest_full.txt "
                              "in %s from a previous (non-resume) run of this same command."
                              % work_dir)
        with open(manifest_full_path) as f:
            manifest_lines = [l.strip() for l in f if l.strip()]
        print("Resuming: reusing cached %s (%d scouting file(s) total)"
              % (matches_path, len(manifest_lines)), flush=True)
    else:
        print("Resolving and staging scouting files for %d dataset(s) (instance=%s)..."
              % (len(args.scouting_dataset), args.dbs_instance), flush=True)
        staged_files = stage_scouting_files(args.scouting_dataset, args.dbs_instance,
                                             work_dir, args.redirector,
                                             max_files=args.max_scouting_files,
                                             max_workers=DEFAULT_STAGE_WORKERS)
        print("  %d file(s) staged to %s/inputs" % (len(staged_files), work_dir), flush=True)

        print("Stage 1: reading event keys from %d scouting file(s)..." % len(staged_files), flush=True)
        scout_df = data_matching.scouting_event_keys(staged_files, max_workers=args.max_workers)
        runs = sorted(int(r) for r in scout_df["run"].unique())
        print("  %d scouting events, %d run(s): %s" % (len(scout_df), len(runs), runs), flush=True)

        print("Stage 2: discovering and matching offline files for %d run(s) across %d dataset(s)..."
              % (len(runs), len(args.offline_datasets)), flush=True)
        offline_files = data_matching.find_offline_files(args.offline_datasets, runs,
                                                           max_workers=args.max_workers)
        print("  %d candidate offline file(s)" % len(offline_files), flush=True)
        matches_df = data_matching.match_offline_files(
            scout_df, offline_files, redirector=args.redirector,
            max_workers=args.max_workers, cache_dir=os.path.join(work_dir, "offline_key_cache"))
        print("  %d matched event(s)" % len(matches_df), flush=True)

        if len(matches_df) == 0:
            raise SystemExit("No matched events found -- nothing to submit. Check --offline-datasets "
                              "cover the right runs/trigger streams.")

        save_matches(matches_path, matches_df)
        print("Wrote %s" % matches_path, flush=True)

        manifest_lines = sorted(matches_df["scout_file"].unique().tolist())
        with open(manifest_full_path, "w") as f:
            f.write("\n".join(manifest_lines) + "\n")
        print("Wrote manifest_full.txt (%d scouting file(s) with >=1 match, out of %d staged)"
              % (len(manifest_lines), len(staged_files)), flush=True)

    if args.dry_run:
        print("--dry-run: stopping before Slurm submission. Inspect %s to sanity-check "
              "before resubmitting without --dry-run." % work_dir)
        return

    # Only resubmit chunks not already sitting on disk -- a no-op superset on
    # a fresh run (nothing exists yet), the actual point on --resume.
    chunks_dir = os.path.join(work_dir, "chunks")
    pending = [f for f in manifest_lines
               if not os.path.exists(os.path.join(chunks_dir, os.path.splitext(os.path.basename(f))[0] + ".npz"))]

    if pending:
        with open(manifest_path, "w") as f:
            f.write("\n".join(pending) + "\n")
        print("Wrote manifest.txt (%d file(s) pending out of %d total)"
              % (len(pending), len(manifest_lines)), flush=True)

        stage_proxy(work_dir)
        print("Staged grid proxy to %s/proxy/x509up" % work_dir, flush=True)

        os.makedirs(os.path.join(work_dir, "logs"), exist_ok=True)
        n_tasks = len(pending)
        cmd = [
            "sbatch", "--account=%s" % args.account, "--partition=%s" % args.partition,
            "--exclude=%s" % args.exclude,
            "--array=0-%d%%%d" % (n_tasks - 1, args.throttle),
            "--mem=%s" % args.mem, "--time=%s" % args.time,
            "--job-name=data_matching_%s" % os.path.basename(work_dir),
            "--output=%s/logs/%%A_%%a.out" % work_dir,
            "--error=%s/logs/%%A_%%a.err" % work_dir,
            "--export=ALL,SCOUT_COLLECTION=%s,MAX_DR=%s" % (args.scout_collection, args.max_dr),
            ARRAY_SCRIPT, work_dir,
        ]
        print("+ %s" % " ".join(cmd), flush=True)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout, flush=True)
        job_id = result.stdout.strip().split()[-1]
        print("Submitted Slurm array job %s (%d tasks)" % (job_id, n_tasks), flush=True)

        poll_job(job_id, n_tasks)
    else:
        print("All %d chunk(s) already present on disk -- nothing to submit." % len(manifest_lines),
              flush=True)

    missing = merge_chunks(work_dir, manifest_lines, args.output)
    if missing:
        print("Re-run with --resume (same --work-dir) -- e.g. with a higher --mem/--time -- "
              "to fill in the %d missing chunk(s)." % len(missing))


if __name__ == "__main__":
    main()
