"""Per-data-taking-period config for the scouting-vs-offline stability study
(submit_data_matching_slurm.py run once per period, results compared across
periods to check whether the truth-inverted SF correction drifts over time).

Only periods with an EXISTING scouting NanoAOD production (jschulte's CRAB
output, DBS instance prod/phys03) are in READY_PERIODS -- data_matching can
only join against a scouting sample that already exists. Everything else
found in central DAS (offline JetMET0/1, Muon0/1 NANOAOD) but with no
matching scouting production is listed in MISSING_SCOUTING_PRODUCTION, so the
gap is explicit rather than silently skipped. Closing that gap means running
ScoutingNanoProduction/submit_scoutingNano.py for those periods first -- a
separate, much larger step (new CRAB tasks, not just a matching job) that
this module deliberately does not do on its own.

Checked 2026-08-27 via dasgoclient (prod/phys03 for scouting, prod/global for
offline); re-verify before relying on this if run much later, since new
scouting production or new offline reprocessing tags can appear.

Sampling choice, found the hard way this session: READY_PERIODS'
scouting_datasets deliberately points at ONE part per era, not every part.
An earlier version used all_scouting_datasets (every part) as the default and
run_data_matching_periods.py's even-file-sampling spread its 60-file cap
across all of them -- for 2024G (11 parts) that touched 53 distinct runs and
ballooned Stage 2 to 3,338 candidate offline files (still running after 110
CPU-minutes and 25GB RSS when killed), vs. 2024H's 2-part run finishing in
~14 min at 16 runs / 686 candidates. One part per era already gave 2025C
(1 part) and 2024H's own part1 a comparably-sized, fast run, so it's the
right default "snapshot" granularity for a stability-vs-time comparison --
use all_scouting_datasets explicitly (via --scouting-dataset on
submit_data_matching_slurm.py directly) only if you deliberately want a
full-era average and are prepared for a much longer Stage 2.
"""

# Offline NANOAOD tag used for every period below: PromptReco-v1 -- verified
# (not assumed) to cover the actual runs present in each scouting sample,
# across all four PDs (JetMET0/1, Muon0/1). Central re-reco tags (e.g.
# MINIv6NANOv15) exist too but their version suffix (-v1, -v2, ...) varies
# per era in a way that isn't safe to hardcode without re-checking each time;
# PromptReco-v1 did not have that problem for any period checked here.
_PDS = ["JetMET0", "JetMET1", "Muon0", "Muon1"]


def _offline_datasets(era, tag="PromptReco-v1"):
    return ["/%s/Run%s-%s/NANOAOD" % (pd, era, tag) for pd in _PDS]


def _all_parts(era, tag, n_parts):
    return [
        "/ScoutingPFRun3/jschulte-ScoutingNano_Data_%s_part%d_%s-"
        "00000000000000000000000000000000/USER" % (era, i, tag)
        for i in range(1, n_parts + 1)
    ]


READY_PERIODS = {
    "2024G": {
        "scouting_datasets": _all_parts("2024G", "v5_jetMatchFix", 1),  # part1 only, see module docstring
        "all_scouting_datasets": _all_parts("2024G", "v5_jetMatchFix", 11),
        "offline_datasets": _offline_datasets("2024G"),
        "run_range": (383811, 383996),  # part1's own run span
    },
    "2024H": {
        "scouting_datasets": _all_parts("2024H", "v5_jetMatchFix", 1),
        "all_scouting_datasets": _all_parts("2024H", "v5_jetMatchFix", 2),
        "offline_datasets": _offline_datasets("2024H"),
        "run_range": (385836, 386010),  # part1's own run span
    },
    "2024I": {
        "scouting_datasets": _all_parts("2024I", "v5_jetMatchFix", 1),
        "all_scouting_datasets": _all_parts("2024I", "v5_jetMatchFix", 4),
        "offline_datasets": _offline_datasets("2024I"),
        "run_range": (386478, 386618),  # part1's own run span
    },
    "2025C": {
        "scouting_datasets": _all_parts("2025C", "v6", 1),
        "all_scouting_datasets": _all_parts("2025C", "v6", 1),  # only 1 part exists
        "offline_datasets": _offline_datasets("2025C"),
        "run_range": (392293, 392295),  # already matched this session
    },
}

# Offline data (JetMET0/1, Muon0/1 NANOAOD) exists centrally for all of these,
# but no scouting NanoAOD has been produced (jschulte-prefixed prod/phys03) --
# confirmed by a full phys03 dataset scan, not just these specific names.
# Producing it is ScoutingNanoProduction/submit_scoutingNano.py's job, not
# this module's.
MISSING_SCOUTING_PRODUCTION = {
    "2024": ["2024A", "2024B", "2024C", "2024D", "2024E", "2024F", "2024J"],
    "2025": ["2025B", "2025D", "2025E", "2025F", "2025G"],
    "2026": ["2026A", "2026B", "2026C", "2026D"],
}
