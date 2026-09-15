"""Match scouting NanoAOD events to existing *centrally-produced* offline
NanoAOD (JetMET0/1, Muon0/1, ...) by (run, luminosityBlock, event), then
dR-match the two jet collections within each matched event.

Scouting and offline reconstruction of data live in physically separate
primary datasets (unlike MC MiniAOD, where both are in one file), so there is
no in-file way to compare scouting jets to offline jets for data. This module
does NOT produce a companion offline sample -- central Run3 NanoAOD already
exists for the offline PDs and covers the same runs as the scouting stream
(verified directly: run 380945, the example run used throughout
ScoutingNanoProduction's standalone configs, is present in
/JetMET0/Run2024D-MINIv6NANOv15-v1/NANOAOD, and streaming that dataset's
run/luminosityBlock/event over xrootd found exact matches against an
already-produced local scouting NanoAOD file for the same run). So the whole
job here is: find the existing files, join by event id, dR-match jets.

Three stages, sized to avoid pulling full jet content for every event in a PD
that's far larger than what's needed:
  1. scouting_event_keys()   -- cheap run/lumi/event + provenance, scouting side
  2. find_offline_files() + match_offline_files() -- DAS-restricted offline
     key scan, joined against the scouting keys
  3. build_comparison_table() -- targeted jet extraction for matched events
     only, then dR matching within each event

Trigger note: DST_PFScouting_JetHT-triggered scouting events should be
matched against JetMET0+JetMET1 (both siblings of the trigger-hash-split PD,
since it isn't confirmed up front which half a given event lands in);
DST_PFScouting_SingleMuon-triggered events against Muon0+Muon1.
"""

import hashlib
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

import awkward as ak
import numpy as np
import pandas as pd
import uproot

from . import io

DEFAULT_REDIRECTOR = "root://cms-xrd-global.cern.ch/"

# Stages 2 and 3 below are I/O-bound (one DAS subprocess call or one xrootd
# open+read per candidate file, each dominated by network/round-trip latency
# rather than CPU) -- both release the GIL while waiting, so a thread pool
# gives near-linear speedup up to the point of saturating the redirector, not
# a process pool. 8 concurrent streams is a conservative default that
# overlaps latency well without hammering a shared site redirector; raise
# --max-workers if the redirector tolerates it (a site-local one, e.g.
# root://cms-xrootd.rcac.purdue.edu/, generally does).
DEFAULT_MAX_WORKERS = 8


def das_query(query):
    """Same tiny dasgoclient wrapper as ScoutingNanoProduction/submit_scoutingNano.py's
    das_query -- duplicated here (not imported cross-package) to keep this
    module runnable standalone, outside the CRAB submission machinery.
    """
    cmd = ["dasgoclient", "-query", query]
    out = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode().strip()
    return [line for line in out.splitlines() if line.strip()]


def scouting_event_keys(files, max_workers=DEFAULT_MAX_WORKERS):
    """Stage 1: run/luminosityBlock/event + (file, local entry) provenance
    for every scouting NanoAOD file, as one pandas DataFrame with columns
    run, luminosityBlock, event, scout_file, scout_entry. These are our own
    already-produced local files, so this is cheap relative to Stages 2-3,
    but threading the reads costs nothing and helps when there are many.
    """
    if not files:
        return pd.DataFrame(columns=["run", "luminosityBlock", "event", "scout_file", "scout_entry"])

    def _read(f):
        return f, io.load_event_keys(f)

    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for f, keys in ex.map(_read, files):
            results[f] = keys

    frames = []
    for f in files:
        keys = results[f]
        n = len(keys["run"])
        frames.append(pd.DataFrame({
            "run": keys["run"],
            "luminosityBlock": keys["luminosityBlock"],
            "event": keys["event"],
            "scout_file": f,
            "scout_entry": np.arange(n),
        }))
    return pd.concat(frames, ignore_index=True)


def find_offline_files(dataset_names, runs, max_workers=DEFAULT_MAX_WORKERS):
    """Stage 2a: DAS file discovery for the offline PD(s), restricted to the
    given runs. dataset_names is a list of full DAS dataset names (pass both
    siblings of a trigger-hash-split PD -- see module docstring). Central
    NanoAOD files are NOT split per-run (a single file spans a range of
    runs), so this only narrows the *candidate* file list; the run/lumi/event
    join in match_offline_files() below still does the real filtering.

    One dasgoclient subprocess call per (dataset, run) pair -- run in a
    thread pool since each call is dominated by DAS's own network round
    trip, not local CPU. Returns a deduplicated, sorted list of LFNs (no
    redirector prefix).
    """
    tasks = [(dataset, run) for dataset in dataset_names for run in runs]

    def _query(task):
        dataset, run = task
        return das_query("file dataset=%s run=%d" % (dataset, int(run)))

    files = set()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for result in ex.map(_query, tasks):
            files.update(result)
    return sorted(files)


def _key_cache_path(cache_dir, lfn):
    # LFNs contain '/' and can be long; hash to a flat, filesystem-safe name.
    h = hashlib.sha1(lfn.encode()).hexdigest()
    return os.path.join(cache_dir, h + ".npz")


def _load_offline_keys(lfn, redirector, cache_dir=None):
    """Read (run, luminosityBlock, event) for one offline file, local-disk
    cached by LFN if cache_dir is given. The key scan (Stage 2) is the part
    that's repeated across every re-run of this pipeline while tuning
    max_dr, the scouting collection, or extending to more scouting files
    covering runs already scanned once -- caching it means only genuinely
    new (dataset, run) combinations pay the xrootd round trip again.
    """
    if cache_dir:
        cache_path = _key_cache_path(cache_dir, lfn)
        if os.path.exists(cache_path):
            d = np.load(cache_path)
            return {"run": d["run"], "luminosityBlock": d["luminosityBlock"], "event": d["event"]}

    url = lfn if lfn.startswith("root:") else redirector + lfn
    keys = io.load_event_keys(url)

    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        np.savez(_key_cache_path(cache_dir, lfn), run=keys["run"],
                 luminosityBlock=keys["luminosityBlock"], event=keys["event"])
    return keys


def match_offline_files(scout_df, offline_files, redirector=DEFAULT_REDIRECTOR,
                         max_workers=DEFAULT_MAX_WORKERS, cache_dir=None):
    """Stage 2b: open each candidate offline file, mask to the runs present
    in scout_df, and inner-join on (run, luminosityBlock, event). This is
    the dominant cost of the whole pipeline (one xrootd open+read per
    candidate file, each mostly network round-trip latency) -- the opens are
    farmed out to a thread pool (max_workers concurrent streams), and each
    file's key scan is optionally cached to cache_dir so a repeat run over
    the same candidate files (e.g. while tuning downstream parameters) skips
    the network entirely. The pandas merge itself stays single-threaded in
    the main thread as each worker's result comes back -- only the I/O is
    parallelized.

    Returns a concatenated DataFrame of matched provenance: run,
    luminosityBlock, event, scout_file, scout_entry, off_file, off_entry
    (off_file is the LFN, without redirector prefix; off_entry is the
    entry's position in the full file, matching how build_comparison_table()
    re-opens it).
    """
    empty = pd.DataFrame(columns=["run", "luminosityBlock", "event",
                                   "scout_file", "scout_entry", "off_file", "off_entry"])
    if len(scout_df) == 0:
        return empty

    runs_of_interest = set(int(r) for r in scout_df["run"].unique())

    def _fetch(lfn):
        return lfn, _load_offline_keys(lfn, redirector, cache_dir)

    matched_frames = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_fetch, lfn): lfn for lfn in offline_files}
        for fut in as_completed(futures):
            lfn = futures[fut]
            try:
                _, keys = fut.result()
            except Exception as exc:
                print("  WARNING: failed to open %s: %s" % (lfn, exc))
                continue
            mask = np.isin(keys["run"], list(runs_of_interest))
            if not mask.any():
                continue
            off_df = pd.DataFrame({
                "run": keys["run"][mask],
                "luminosityBlock": keys["luminosityBlock"][mask],
                "event": keys["event"][mask],
                "off_file": lfn,
                "off_entry": np.nonzero(mask)[0],
            })
            merged = scout_df.merge(off_df, on=["run", "luminosityBlock", "event"], how="inner")
            if len(merged):
                matched_frames.append(merged)

    if not matched_frames:
        return empty
    return pd.concat(matched_frames, ignore_index=True)


def _delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def dr_match_jets(eta_a, phi_a, eta_b, phi_b, max_dr=0.4):
    """Greedy nearest-neighbor dR matching between two jet collections from
    the SAME event (1D arrays, one entry per jet). Closest pairs are
    assigned first and each jet is used in at most one pair -- the standard
    approach for matching two independently-clustered jet collections (no
    pre-existing dR matcher exists anywhere in this repo: response.py's
    matching relies entirely on MC's pre-computed genJetIdx, which has no
    cross-dataset equivalent here). Returns a list of (idx_a, idx_b, dr)
    tuples, sorted by ascending dr.
    """
    eta_a = np.asarray(eta_a, dtype=float)
    phi_a = np.asarray(phi_a, dtype=float)
    eta_b = np.asarray(eta_b, dtype=float)
    phi_b = np.asarray(phi_b, dtype=float)
    na, nb = len(eta_a), len(eta_b)
    if na == 0 or nb == 0:
        return []

    deta = eta_a[:, None] - eta_b[None, :]
    dphi = _delta_phi(phi_a[:, None], phi_b[None, :])
    dr = np.sqrt(deta ** 2 + dphi ** 2)

    ia, ib = np.nonzero(dr < max_dr)
    candidates = sorted(zip(dr[ia, ib], ia.tolist(), ib.tolist()))

    used_a, used_b = set(), set()
    matches = []
    for d, i, j in candidates:
        if i in used_a or j in used_b:
            continue
        used_a.add(i)
        used_b.add(j)
        matches.append((i, j, float(d)))
    return matches


_SCOUT_FIELDS = ["pt", "eta", "phi", "mass", "rawFactor"]
_OFF_FIELDS = ["pt", "eta", "phi", "mass", "rawFactor", "area"]


def build_comparison_table(matches_df, scout_collection="ScoutingPFJetRecluster2",
                            offline_prefix="Jet", scout_fields=None, off_fields=None,
                            redirector=DEFAULT_REDIRECTOR, max_dr=0.4,
                            max_workers=DEFAULT_MAX_WORKERS):
    """Stage 3: for the event pairs found by match_offline_files(), pull just
    the matched entries' jet branches (scouting: scout_collection, e.g.
    "ScoutingPFJetRecluster2"/"ScoutingPFJetReclusterCHS" for AK4 or
    "ScoutingFatPFJetRecluster" for AK8 -- see io.BASELINE_COLLECTIONS/
    BASELINE_COLLECTIONS_AK8; offline: offline_prefix="Jet" (default, AK4) or
    "FatJet" (AK8, official NanoAOD's standard name for the PUPPI AK8
    collection)), dR-match the two jet collections within each event, and
    return one flat table (dict of numpy arrays, one row per matched jet
    pair) shaped like response.py's response_table() output so it plugs into
    the same _binned_stats-based plotting.

    scout_fields/off_fields default to the AK4 field lists (_SCOUT_FIELDS/
    _OFF_FIELDS); pass e.g. scout_fields=off_fields=["pt","eta","phi","mass",
    "area"] for AK8 -- the scouting AK8 collection has no rawFactor branch
    (unlike AK4), so the AK4 defaults' "rawFactor" would KeyError there.

    Files are opened one at a time, in (scout_file, off_file) group order --
    NOT farmed out to a thread pool with all results cached up front (unlike
    match_offline_files()'s Stage 2) -- see the loop below for why: caching
    every distinct file's full column set at once does not scale to eras
    with many candidate offline files. max_workers is accepted but unused
    here (kept for call-site compatibility with the other Stage functions).
    """
    if scout_fields is None:
        scout_fields = _SCOUT_FIELDS
    if off_fields is None:
        off_fields = _OFF_FIELDS

    out = {"run": [], "luminosityBlock": [], "event": [], "dr": []}
    for f in scout_fields:
        out["scout_%s" % f] = []
    for f in off_fields:
        out["off_%s" % f] = []

    if len(matches_df) == 0:
        return {k: np.asarray(v) for k, v in out.items()}

    scout_branches = ["%s_%s" % (scout_collection, f) for f in scout_fields]
    off_branches = ["%s_%s" % (offline_prefix, f) for f in off_fields]

    def _open_scout(f):
        return uproot.open(f)["Events"].arrays(scout_branches, library="ak")

    def _open_off(f):
        url = f if f.startswith("root:") else redirector + f
        return uproot.open(url)["Events"].arrays(off_branches, library="ak")

    # Process one (scout_file, off_file) group at a time, keeping at most one
    # scout file's and one offline file's columns in memory at once -- NOT
    # every distinct file up front via a thread pool (the previous approach
    # here). That was fine for a small matched-event set (e.g. 2025C's ~600k
    # pairs, a few dozen files), but some eras' matches_df spans hundreds of
    # distinct offline files, each fully materialized (every event, not just
    # the matched local entries) and ALL held simultaneously for the whole
    # loop -- confirmed via dmesg to OOM-kill the process at >80GB RSS for
    # 2024G. groupby's default sort keeps repeats of the same file adjacent
    # for the common case (same PD file matching many consecutive events),
    # so this rarely reopens a file it just processed; max_workers/threading
    # is deliberately given up here in exchange for bounded memory.
    current_scout_file, scout_cols = None, None
    current_off_file, off_cols = None, None

    for (scout_file, off_file), group in matches_df.groupby(["scout_file", "off_file"]):
        if scout_file != current_scout_file:
            s_arr = _open_scout(scout_file)
            scout_cols = {f: ak.to_list(s_arr["%s_%s" % (scout_collection, f)]) for f in scout_fields}
            current_scout_file = scout_file
        if off_file != current_off_file:
            o_arr = _open_off(off_file)
            off_cols = {f: ak.to_list(o_arr["%s_%s" % (offline_prefix, f)]) for f in off_fields}
            current_off_file = off_file

        s_cols, o_cols = scout_cols, off_cols

        for row in group.itertuples():
            s_jets = {f: np.asarray(s_cols[f][row.scout_entry]) for f in scout_fields}
            o_jets = {f: np.asarray(o_cols[f][row.off_entry]) for f in off_fields}

            pairs = dr_match_jets(s_jets["eta"], s_jets["phi"], o_jets["eta"], o_jets["phi"], max_dr=max_dr)
            for i, j, d in pairs:
                out["run"].append(row.run)
                out["luminosityBlock"].append(row.luminosityBlock)
                out["event"].append(row.event)
                out["dr"].append(d)
                for f in scout_fields:
                    out["scout_%s" % f].append(s_jets[f][i])
                for f in off_fields:
                    out["off_%s" % f].append(o_jets[f][j])

    return {k: np.asarray(v) for k, v in out.items()}


_MC_COMMON_FIELDS = ["pt", "eta", "phi", "mass", "rawFactor"]


def nearest_dr(eta_a, phi_a, eta_b, phi_b):
    """For each object in (eta_a, phi_a), the dR to its closest object in
    (eta_b, phi_b) -- NOT exclusive/greedy like dr_match_jets (a gen tau can
    be the "nearest" object to more than one jet; that's fine, we only care
    about proximity here, not a one-to-one assignment). Returns an array of
    length len(eta_a); 99.0 (sentinel, safely outside any realistic max_dr
    cut) where eta_b is empty.
    """
    eta_a = np.asarray(eta_a, dtype=float)
    phi_a = np.asarray(phi_a, dtype=float)
    eta_b = np.asarray(eta_b, dtype=float)
    phi_b = np.asarray(phi_b, dtype=float)
    if len(eta_b) == 0:
        return np.full(len(eta_a), 99.0)
    deta = eta_a[:, None] - eta_b[None, :]
    dphi = _delta_phi(phi_a[:, None], phi_b[None, :])
    dr = np.sqrt(deta ** 2 + dphi ** 2)
    return dr.min(axis=1)


def build_comparison_table_mc(files, scout_collection="ScoutingPFJetRecluster2",
                               offline_prefix="OfflineJet", fields=None,
                               scout_fields=None, off_fields=None, max_dr=0.4,
                               gentau_collection=None):
    """MC equivalent of build_comparison_table(), for files produced with
    customiseScoutingNanoWithOfflineJets() (scoutingToMiniAODDerivedCollections_
    cff.py) wired into the production cfg -- see scoutingnano_mc_standalone2.py.
    There, scouting and offline jets already live in the same event of the
    same file (MC MiniAOD has both), so Stages 1-2 of the data-side pipeline
    (event-key extraction, DAS file discovery, run/lumi/event join) don't
    apply here at all; only Stage 3's dR-matching does, applied directly
    event-by-event within each file.

    offline_prefix="OfflineFatJet" (with scout_collection=
    "ScoutingFatPFJetRecluster2" or a PUPPI AK8 variant name) does the AK8
    comparison instead. fields (applied to BOTH sides) defaults to
    _MC_COMMON_FIELDS, the set present on both OfflineJet/OfflineFatJet and
    the scouting jet tables (see the branch-name-parity note in
    customiseScoutingNanoWithOfflineJets's docstring). For fields that only
    exist on one side -- e.g. hadronFlavour/partonFlavour, offline-only truth
    info with no scouting equivalent -- pass scout_fields/off_fields
    separately instead (each defaults to `fields` if not given); scout_fields
    must still include at least eta/phi (needed for dR matching).

    gentau_collection, if given (e.g. "GenVisTau", see
    customiseScoutingNanoWithGenVisTau() in scoutingToMiniAODDerivedCollections_
    cff.py -- MC-only, requires the production cfg to have called that
    customisation), adds one extra output column "off_genVisTau_dr": the dR
    from each matched offline jet to the closest GenVisTau in the same event
    (99.0 sentinel if the event has none), via nearest_dr() -- a genuine
    hadronic-tau-jet tag, since GenVisTau is built from gen-level visible tau
    decay products, not a reconstructed discriminator. This is NOT a 1:1
    dr_match_jets() assignment: a single gen tau can legitimately be the
    closest object to more than one reconstructed jet, and we only need
    proximity here, not exclusivity.

    Returns a flat table (dict of numpy arrays) with the same shape as
    build_comparison_table()'s output (scout_<field>, off_<field>, dr, run,
    luminosityBlock, event), so response.response_vs_pt_data() works
    unchanged on either.
    """
    if fields is None:
        fields = _MC_COMMON_FIELDS
    if scout_fields is None:
        scout_fields = fields
    if off_fields is None:
        off_fields = fields

    out = {"run": [], "luminosityBlock": [], "event": [], "dr": []}
    for f in scout_fields:
        out["scout_%s" % f] = []
    for f in off_fields:
        out["off_%s" % f] = []
    if gentau_collection:
        out["off_genVisTau_dr"] = []

    scout_branches = ["%s_%s" % (scout_collection, f) for f in scout_fields]
    off_branches = ["%s_%s" % (offline_prefix, f) for f in off_fields]
    gentau_branches = (["%s_eta" % gentau_collection, "%s_phi" % gentau_collection]
                        if gentau_collection else [])

    for fpath in files:
        tree = uproot.open(fpath)["Events"]
        arrays = tree.arrays(
            scout_branches + off_branches + gentau_branches + ["run", "luminosityBlock", "event"],
            library="ak")
        n = len(arrays)

        # Materialize each column to plain python lists once per file (fast,
        # done in C by awkward) instead of re-slicing the awkward record
        # array per event below -- doing the latter for every (event, field)
        # pair is what made this loop take ~4-5 min per 100 MB of input.
        runs = ak.to_list(arrays["run"])
        lumis = ak.to_list(arrays["luminosityBlock"])
        events = ak.to_list(arrays["event"])
        s_cols = {f: ak.to_list(arrays["%s_%s" % (scout_collection, f)]) for f in scout_fields}
        o_cols = {f: ak.to_list(arrays["%s_%s" % (offline_prefix, f)]) for f in off_fields}
        if gentau_collection:
            gt_eta = ak.to_list(arrays["%s_eta" % gentau_collection])
            gt_phi = ak.to_list(arrays["%s_phi" % gentau_collection])

        for i in range(n):
            s_jets = {f: np.asarray(s_cols[f][i]) for f in scout_fields}
            o_jets = {f: np.asarray(o_cols[f][i]) for f in off_fields}

            pairs = dr_match_jets(s_jets["eta"], s_jets["phi"], o_jets["eta"], o_jets["phi"], max_dr=max_dr)
            if gentau_collection and pairs:
                dr_to_tau = nearest_dr(o_jets["eta"], o_jets["phi"], gt_eta[i], gt_phi[i])
            for a, b, d in pairs:
                out["run"].append(runs[i])
                out["luminosityBlock"].append(lumis[i])
                out["event"].append(events[i])
                out["dr"].append(d)
                for f in scout_fields:
                    out["scout_%s" % f].append(s_jets[f][a])
                for f in off_fields:
                    out["off_%s" % f].append(o_jets[f][b])
                if gentau_collection:
                    out["off_genVisTau_dr"].append(dr_to_tau[b])

    return {k: np.asarray(v) for k, v in out.items()}
