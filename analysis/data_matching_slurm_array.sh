#!/bin/bash
# Per-task worker for the Slurm array submitted by submit_data_matching_slurm.py.
# One line of $WORK/manifest.txt (the staged scout_file path) per array task.
#
# Two gotchas this follows (carried over from this package's original home in
# ScoutingPuppiCalibration/Calibration/analysis -- see that repo's
# memory/scouting_puppi_recalibration.md for how these were found):
#   1. /eos/purdue is NOT mounted on hammer-nodes compute nodes -- $WORK/manifest.txt
#      must list already-staged /depot paths (submit_data_matching_slurm.py
#      stages them via cp before submitting), never /eos/purdue/... paths.
#   2. The default $X509_USER_PROXY (/tmp/x509up_u<uid>) lives on the login
#      node's node-local /tmp, invisible to compute nodes -- the driver
#      copies the proxy to $WORK/proxy/ and this script points at that copy.
set -e

# This package has no PyROOT/CMSSW dependency at all (pure uproot/awkward/
# pandas/correctionlib) -- CMSSW_SRC is used ONLY as a convenient way to get
# a working python3 + uproot/awkward/correctionlib stack via `scram runtime`
# on this cluster, where that's the easiest pre-built environment available.
# Point it at any CMSSW checkout with those externals, or replace the
# `source .../scram runtime` block below with your own venv/conda activate
# if you'd rather not depend on a CMSSW area at all.
CMSSW_SRC="${CMSSW_SRC:-/depot/cms/private/users/schul105/VVV/deepntuples/production/CMSSW_16_1_0_pre4/src}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

WORK="$1"
# MANIFEST_FILE/CHUNKS_DIR let a caller shard a DIFFERENT Stage-3 variant
# (different scout_collection/offline_prefix/fields, e.g. adding tagger-score
# branches) against the SAME cached matches.npz without colliding with
# another variant's manifest.txt/chunks/ already sitting in this work-dir --
# both default to the original submit_data_matching_slurm.py paths so this
# script is unchanged for that caller.
MANIFEST="${MANIFEST_FILE:-$WORK/manifest.txt}"
CHUNKS_DIR="${CHUNKS_DIR:-$WORK/chunks}"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MANIFEST")
SCOUT_FILE="$LINE"

source /cvmfs/cms.cern.ch/cmsset_default.sh
eval `cd "$CMSSW_SRC" && scram runtime -sh`

export X509_USER_PROXY="$WORK/proxy/x509up"

BASE=$(basename "$SCOUT_FILE" .root)
mkdir -p "$CHUNKS_DIR"
OUT="$CHUNKS_DIR/${BASE}.npz"

python3 "$SCRIPT_DIR/data_matching_stage3_chunk.py" \
  --matches "$WORK/matches.npz" \
  --scout-file "$SCOUT_FILE" \
  --scout-collection "$SCOUT_COLLECTION" \
  --offline-prefix "${OFFLINE_PREFIX:-Jet}" \
  ${FIELDS:+--fields $FIELDS} \
  ${OFF_FIELDS:+--off-fields $OFF_FIELDS} \
  --max-dr "$MAX_DR" \
  --outfile "$OUT"
