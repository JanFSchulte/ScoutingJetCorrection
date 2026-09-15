"""Loading helpers for the scouting-vs-offline jet comparison pipeline.

This is a trimmed copy of the io.py module shared with the (separate)
ScoutingPuppiCalibration/Calibration PUPPI-recalibration package: only the
pieces data_matching.py actually uses are kept here (BASELINE_COLLECTIONS/
BASELINE_COLLECTIONS_AK8, load_event_keys). The PUPPI-recalibration-specific
loaders (load_jets, load_genjets, load_candidates, matched_response,
matched_mass, puppi_collection_name*) are NOT needed by this package and
were deliberately left out rather than copied unused.
"""

import uproot

# branch-name prefix (NanoAOD table "name=") -> module label used for the
# jet collection, so callers can refer to either scouting baseline by the
# same short key.
BASELINE_COLLECTIONS = {
    "plain": "ScoutingPFJetRecluster2",
    "chs": "ScoutingPFJetReclusterCHS",
}

# AK8: plain-only baseline (no CHS-AK8 collection exists).
BASELINE_COLLECTIONS_AK8 = {
    "plain": "ScoutingFatPFJetRecluster2",
}


def load_event_keys(file_or_url, branches=("run", "luminosityBlock", "event")):
    """Read event-id branches from a single file (local path or xrootd URL),
    in on-disk order, so the returned arrays' positions double as local entry
    indices. Used by data_matching.py to join scouting and offline files by
    (run, luminosityBlock, event) without touching any jet branches first.
    """
    tree = uproot.open(file_or_url)["Events"]
    arrays = tree.arrays(list(branches), library="np")
    return {b: arrays[b] for b in branches}
