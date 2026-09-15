#!/usr/bin/env python3
"""Shard Stage 3 (data_matching.build_comparison_table) across a Slurm array,
reusing an already-cached Stage 1-2 work-dir (matches.npz + manifest_full.txt,
written by a prior non---resume run of submit_data_matching_slurm.py).

submit_data_matching_slurm.py already does exactly this sharding (one array
task per scouting file, via data_matching_slurm_array.sh ->
data_matching_stage3_chunk.py) for its own default field set, but its
--resume path reuses the SAME work-dir/chunks/ and work-dir/manifest.txt --
fine for re-running the identical variant, but it would collide with (or
silently reuse stale) chunks for a DIFFERENT scout_collection/offline_prefix/
field list (e.g. adding an offline tagger-score branch to re-derive a
b-tag-category correction against the same matched events). This script
shards a distinct variant into its own --chunks-subdir/--manifest-name so it
never touches another variant's cache, then merges into --output -- Stages
1-2 are NOT repeated.

Usage:
    python run_stage3_slurm.py \\
        --work-dir results/data_matching_periods/2024G \\
        --variant tagger_ak4 \\
        --scout-collection ScoutingPFJetRecluster2 --offline-prefix Jet \\
        --off-fields pt eta phi mass rawFactor area btagUParTAK4B btagUParTAK4TauVJet \\
        --output results/data_matching_periods/2024G_tagger_ak4.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import submit_data_matching_slurm as sms

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ARRAY_SCRIPT = os.path.join(THIS_DIR, "data_matching_slurm_array.sh")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", required=True,
        help="Existing work-dir from a prior submit_data_matching_slurm.py run -- must "
             "already contain matches.npz, manifest_full.txt, and proxy/ (staged there).")
    parser.add_argument("--variant", required=True,
        help="Short tag (e.g. tagger_ak4, ak4chs, tagger_ak8) identifying this Stage-3 "
             "field/collection variant -- used to namespace manifest/chunks so this run "
             "never collides with another variant's cache in the same work-dir.")
    parser.add_argument("--scout-collection", default="ScoutingPFJetRecluster2")
    parser.add_argument("--offline-prefix", default="Jet")
    parser.add_argument("--fields", nargs="+", default=None,
        help="Scout-side field list (offline side too, unless --off-fields given).")
    parser.add_argument("--off-fields", nargs="+", default=None,
        help="Offline-side field list, overriding --fields for that side only.")
    parser.add_argument("--max-dr", type=float, default=0.4)
    parser.add_argument("--output", required=True)
    parser.add_argument("--account", default=sms.DEFAULT_ACCOUNT)
    parser.add_argument("--partition", default=sms.DEFAULT_PARTITION)
    parser.add_argument("--exclude", default=sms.DEFAULT_EXCLUDE)
    parser.add_argument("--mem", default="4G")
    parser.add_argument("--time", default="30")
    parser.add_argument("--throttle", type=int, default=30,
        help="Max concurrent array tasks (default 30, conservative -- keeps several "
             "variants run one-after-another from ever combining to more than a few "
             "dozen simultaneous xrootd connections to the shared offline redirector).")
    args = parser.parse_args()

    work_dir = os.path.abspath(args.work_dir)
    matches_path = os.path.join(work_dir, "matches.npz")
    manifest_full_path = os.path.join(work_dir, "manifest_full.txt")
    if not (os.path.exists(matches_path) and os.path.exists(manifest_full_path)):
        raise SystemExit("%s is missing matches.npz/manifest_full.txt -- run "
                          "submit_data_matching_slurm.py (Stages 1-2) for this era first."
                          % work_dir)

    with open(manifest_full_path) as f:
        manifest_lines = [l.strip() for l in f if l.strip()]
    print("Reusing cached Stage 1-2 match table: %d scouting file(s)" % len(manifest_lines), flush=True)

    chunks_dir = os.path.join(work_dir, "chunks_%s" % args.variant)
    manifest_path = os.path.join(work_dir, "manifest_%s.txt" % args.variant)
    os.makedirs(chunks_dir, exist_ok=True)

    pending = [f for f in manifest_lines
               if not os.path.exists(os.path.join(chunks_dir, os.path.splitext(os.path.basename(f))[0] + ".npz"))]

    if pending:
        with open(manifest_path, "w") as f:
            f.write("\n".join(pending) + "\n")
        print("Wrote %s (%d file(s) pending out of %d total)"
              % (manifest_path, len(pending), len(manifest_lines)), flush=True)

        proxy_path = sms.stage_proxy(work_dir)
        print("Staged grid proxy to %s" % proxy_path, flush=True)

        os.makedirs(os.path.join(work_dir, "logs"), exist_ok=True)
        n_tasks = len(pending)
        export_vars = ["ALL",
                       "SCOUT_COLLECTION=%s" % args.scout_collection,
                       "OFFLINE_PREFIX=%s" % args.offline_prefix,
                       "MAX_DR=%s" % args.max_dr,
                       "MANIFEST_FILE=%s" % manifest_path,
                       "CHUNKS_DIR=%s" % chunks_dir]
        if args.fields:
            export_vars.append("FIELDS=%s" % " ".join(args.fields))
        if args.off_fields:
            export_vars.append("OFF_FIELDS=%s" % " ".join(args.off_fields))

        cmd = [
            "sbatch", "--account=%s" % args.account, "--partition=%s" % args.partition,
            "--exclude=%s" % args.exclude,
            "--array=0-%d%%%d" % (n_tasks - 1, args.throttle),
            "--mem=%s" % args.mem, "--time=%s" % args.time,
            "--job-name=stage3_%s_%s" % (os.path.basename(work_dir), args.variant),
            "--output=%s/logs/%%A_%%a.out" % work_dir,
            "--error=%s/logs/%%A_%%a.err" % work_dir,
            "--export=%s" % ",".join(export_vars),
            ARRAY_SCRIPT, work_dir,
        ]
        print("+ %s" % " ".join(cmd), flush=True)
        import subprocess
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(result.stdout, flush=True)
        job_id = result.stdout.strip().split()[-1]
        print("Submitted Slurm array job %s (%d tasks)" % (job_id, n_tasks), flush=True)

        sms.poll_job(job_id, n_tasks)
    else:
        print("All %d chunk(s) already present in %s -- nothing to submit."
              % (len(manifest_lines), chunks_dir), flush=True)

    missing = sms.merge_chunks_from_dir(chunks_dir, manifest_lines, args.output)
    if missing:
        print("Re-run this same command (same --work-dir/--variant) -- e.g. with a higher "
              "--mem/--time -- to fill in the %d missing chunk(s)." % len(missing))


if __name__ == "__main__":
    main()
