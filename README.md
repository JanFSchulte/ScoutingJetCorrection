# ScoutingJetCorrection

Derives and packages a pT/eta-dependent correction that removes the
response bias between CMS Run 3 **scouting** jets (online reconstruction)
and **offline** jets (full PF+PUPPI reconstruction, fully JEC-corrected),
in both simulation and real collision data, and ships the result as a
[correctionlib](https://github.com/cms-nanoAOD/correctionlib) `CorrectionSet`
usable from coffea/awkward or RDataFrame.

Extracted from the `ScoutingPuppiCalibration/Calibration` CMSSW package,
where this started as one strand of a broader scouting-PUPPI R&D effort.
This code has **no CMSSW/PyROOT/RDataFrame dependency at runtime** — it is
pure `uproot`/`awkward`/`pandas`/`numpy`/`correctionlib`, and can run in any
Python environment with those packages installed (see `requirements.txt`).
`analysis/data_matching_slurm_array.sh` sources a CMSSW `scram runtime`
environment purely as a convenient way to get that stack on this cluster —
swap it for your own venv/conda activate if you don't want that dependency
either.

## What's here

- **`analysis/jet_correction.py`** — core correction machinery: derive a
  truth-binned response curve `R(true_pt, eta) = median(scout_pt/off_pt)`
  from a matched comparison table, and invert it per-jet (bisection on
  `raw_pt = true_pt * R(true_pt)`, with flat extrapolation of `R` — not the
  output pt — outside the calibrated range) to correct a raw scouting jet's
  pt.
- **`analysis/data_matching.py`** — matches scouting NanoAOD events to
  centrally-produced offline NanoAOD (JetMET0/1, Muon0/1) by
  `(run, luminosityBlock, event)` (data has no shared file between the two),
  then dR-matches jets within each matched event.
- **`analysis/run_mc_offline_comparison.py`** — the MC-side equivalent:
  scouting and offline jets already live in the same file (via a custom
  `OfflineJet`/`OfflineFatJet` table), so this dR-matches directly, no
  cross-file join needed.
- **`analysis/submit_data_matching_slurm.py`** / **`run_stage3_slurm.py`** /
  **`data_matching_slurm_array.sh`** / **`data_matching_stage3_chunk.py`** —
  shard the (expensive, xrootd-bound) jet-extraction stage across a Slurm
  array, one task per scouting file.
- **`analysis/run_flavor_correction.py`** — derives a *dedicated* per-category
  correction (b/light truth flavour or offline b-tag score, hadronic-tau
  match) alongside the inclusive one, since an inclusive map fit mostly to
  the majority species can leave a real residual on minority flavours.
- **`analysis/build_correctionlib_set.py`** / **`correctionlib_export.py`** —
  package every derived response curve (MC and data, every jet collection,
  every category) into one `correctionlib` `CorrectionSet`.
- **`analysis/correctionlib_usage_example.py`** — how to apply the packaged
  corrections in a coffea/awkward analysis (or RDataFrame via
  correctionlib's C++ API — see the docstring at the bottom of that file).
- **`analysis/plot_*.py`** / **`run_offline_comparison_plots.py`** —
  response/resolution-vs-pt/eta illustration plots.
- **`results/scoutingPUPPI_corrections.json.gz`** — the packaged
  `CorrectionSet` as currently derived (9 `Correction`s: AK4 plain, AK4 CHS,
  AK8, each for MC and for two real-data periods).

## What's *not* here

The multi-hundred-MB matched-jet comparison tables (`.npz`, one row per
dR-matched scouting/offline jet pair) that every correction is derived from
are **not** committed here — they're regenerable via the data-matching
pipeline against CMS DAS/xrootd (requires a valid grid proxy), but too large
for git. See `analysis/build_correctionlib_set.py`'s `GROUPS` list for the
exact table path each shipped correction was built from, and
`analysis/data_matching_periods.py` for the per-era dataset config used to
regenerate the data-side ones.

## Usage

```bash
pip install -r requirements.txt

# Regenerate the packaged CorrectionSet from already-derived response curves
# and comparison tables (paths in build_correctionlib_set.py's GROUPS):
python analysis/build_correctionlib_set.py --output results/scoutingPUPPI_corrections.json.gz

# Apply it in an analysis:
python -c "
import correctionlib
cset = correctionlib.CorrectionSet.from_file('results/scoutingPUPPI_corrections.json.gz')
sf = cset['AK4_CHS_Data2024'].evaluate('btag', 0.5, 60.0)
print(sf)
"
```

See `analysis/correctionlib_usage_example.py` for the full coffea/awkward
and RDataFrame usage patterns, and `analysis/run_data_matching.py` /
`analysis/run_mc_offline_comparison.py` for how a new comparison table is
built from scratch.
