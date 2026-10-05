# Reproducibility

This document describes the exact data, splits, and commands needed to reproduce LocAlign's
training data, evaluation numbers, and (for the data pipeline) the pair manifests themselves.
It covers the released pair manifests, split labels, ligand atom
mappings, exclusion logs, the CATH version used, software commands, random seeds, and
evaluation scripts.

## What is released in this repo

| Artifact | Path |
|---|---|
| Homology-safe split (main model) | `datasets/csv_files/homology_25_10/{train,test}.csv` |
| Ligand-grouped split | `datasets/csv_files/ligand_25_10/{train,test}.csv` |
| Baseline-comparison manifests (same pairs + precomputed baseline results) | `datasets/csv_files/{homology,ligand}_25_10/test_baseline*.csv` |
| Ligand atom-name correspondence (evaluation pairs) | `datasets/ligand_atom_mappings/{homology,ligand}_25_10/test.jsonl` |
| Data pipeline scripts | `miners/parsers/biolip_reader.ipynb`, `miners/scripts/align.py`, `aligner_dl/datasets/utils/split_dataset.py` |
| Ligand-mapping generation script | `scripts/build_ligand_atom_mappings.py` |
| Partition statistics & leakage-check script | `scripts/report_partition_stats.py` |
| Evaluation scripts | `scripts/offline_metrics.py`, `scripts/offline_metrics_apo.py` |
| Training entrypoint | `aligner_dl/trainers/lightning_trainer.py` |

**Released checkpoints**: the homology-split model `checkpoints/baseline/epoch=9-step=87120.ckpt`
and the ligand-split model `checkpoints/baseline-ligand-split/epoch=8-step=97191.ckpt`, each with
its `model_config.yaml`, `dataset_config.yaml` and pLRMSD/eLRMSD calibration models, which
`scripts/inference.py` loads from the checkpoint's directory by default.

**Not released**: the ablation checkpoints;
raw PDB/mmCIF structures, the raw BioLiP text dump, and the full CATH classification files
(all large, third-party, publicly redownloadable — see version pins below instead of
redistributing them); the `train.csv` splits' row-level ligand atom mappings (~80k-100k pairs —
these are recomputed on the fly by the training dataloader itself, see below, so materializing
them here would just be a slow, redundant copy of what running the code already does).
`train.csv` and `test.csv`/`test_baseline*.csv` manifests themselves *are* released in full.

## Local setup

All paths are resolved from `aligner_dl/utils/constants.py`. Tracked files (`datasets/`,
`example_inputs/`, configs) are read from the code's own checkout; untracked data (`checkpoints/`,
`ablation_dfs/`, `results/`) is found from the repo root (or the main checkout when run from a
git worktree); data and tools outside the repo default to sibling directories of the repo
(`../scannet_2212`, `../ligands_25_10_2025`, ...). If your layout differs, copy `.env.example` to
`.env` and set the variables you need; it is loaded automatically, and variables set in the shell or
job take precedence. `LOCALIGN_DATA_ROOT` relocates the untracked data and must be set in the real environment.

## Pipeline stages and exact commands

No single top-level pipeline script exists; stages are run in this order:

**Stage A — raw data ingest.** BioLiP non-redundant interaction data and the CATH domain
classification (see [Version pins](#version-pins)) are read directly by
`miners/parsers/biolip_reader.ipynb`.

**Stage B — candidate filtering & CATH-degree pairing** (`miners/parsers/biolip_reader.ipynb`).
Produces `calculated_cath/{N}_25_10_2025.csv`, the candidate pair pool before per-pair structural
validation. Filters applied, in order (see [Exclusion logs](#exclusion-logs)):
1. Drop ligands that bind only one PDB chain (`ligand_id` singletons).
2. Drop non-small-molecule ligand types (`DNA`/`RNA`/`PEPTIDE`/`NONE` in `ligand_id`).
3. Drop ligands with < 10 heavy atoms (via RDKit `GetNumHeavyAtoms()` on the BioLiP `ligand.tsv.gz`
   SMILES).
4. Cap candidate pair combinations at 10,000 per ligand (random subsample when more chains
   share a ligand).
5. Drop pairs with CATH degree ≥ 5 (`compare_rows` structural-classification distance between
   the two chains' CATH domains).

**Stage C — per-pair structural alignment & validation** (`miners/scripts/align.py` →
`miners/parsers/pair.py::parse_protein_pairs` → `miners/utils/process_pair.py::align_pair` →
`miners/objects/protein_pair.py::find_ligand_transformations`, which calls
`validate_ligand_pair`). Pairs that fail are kept with a non-empty `failure_message` and are
dropped at Stage D (now logged, see below). Validation criteria:
- Ligand atom-count ratio between the two structures > 1.2 → rejected.
- Ligand atom-name overlap between the two structures < 0.8 → rejected.
- Zero, or more than one, ligand-binding residue mapping found → rejected.

**Stage D — train/test split** (`aligner_dl/datasets/utils/split_dataset.py::split_csv`):
```bash
# Homology-safe split (main model, aligner_dl/configs/loc_align.yaml)
python -m aligner_dl.datasets.utils.split_dataset \
    --input_csv calculated_cath/<N>_25_10_2025.csv \
    --output_dir datasets/csv_files/homology_25_10 \
    --test_size 0.2 --val_size 0.1 --mmseq_id_threshold 0.5

# Ligand-grouped split (no ligand shared across train/test)
python -m aligner_dl.datasets.utils.split_dataset \
    --input_csv calculated_cath/<N>_25_10_2025.csv \
    --output_dir datasets/csv_files/ligand_25_10 \
    --test_size 0.2 --val_size 0.1 --group_by_ligand
```
This stage drops exact-duplicate `(tar_protein, src_protein)` pairs and any row with a
non-empty `failure_message` from Stage C, and (as of this change) writes both to
`<output_dir>/excluded_pairs.csv` with an `exclusion_reason` column, instead of only printing
row counts.

`homology_25_10`/`ligand_25_10` = split-generation date (25 Oct 2025), not a threshold value.
`split_dataset.py` internally draws a "val" and a "test" partition. In the released splits the
internal "test" partition was empty, and the "val" partition is the held-out set on which all
reported numbers are computed. That set was originally saved as `val.csv` and has been renamed
`test.csv` (with all derived files, e.g. `val_baseline*.csv` → `test_baseline*.csv`), because
it served as the test set. `train.csv` was not re-split. See
[Training protocol](#training-protocol).

## Exclusion logs

Historically, every filter above only printed a before/after row count — no filtered-out rows
were persisted. This PR makes that persistent going forward:
- `miners/parsers/biolip_reader.ipynb` writes each Stage B filter's dropped rows to
  `calculated_cath/exclusion_logs/{01_singleton_ligand,02_ligand_type,03_heavy_atom_count,04_cath_degree}_excluded.csv`.
- `aligner_dl/datasets/utils/split_dataset.py` writes Stage D's dropped rows (duplicates +
  Stage C failures) to `<output_dir>/excluded_pairs.csv`.

We did not rerun the full historical pipeline to reconstruct the exact exclusion list behind
the already-committed `train`/`test` manifests (this would require re-touching the raw
BioLiP/CATH/structure data at real compute cost); the manifests themselves are released
verbatim instead, and new/rerun pipeline executions will now produce exclusion logs by
construction.

## Ligand atom mapping

`ligand_rmsd` is computed over the shared atom names between a pair's source and target
ligand instances. This correspondence was previously only computed on the fly during
data loading (`aligner_dl/datasets/scannet_dataset.py::PairDataset._read_ligand` and
`__getitem__`, ~lines 125-132) by intersecting PDB atom-name strings read from per-pair
ligand-only PDB files (`{LIGAND_DIR}/{ligand_id}/{protein+chain}_ligand.pdb`, PDB Chemical
Component Dictionary atom naming). `scripts/build_ligand_atom_mappings.py` reimplements
that same logic to materialize it explicitly for the evaluation splits:
```bash
python scripts/build_ligand_atom_mappings.py \
    --manifest datasets/csv_files/homology_25_10/test.csv \
    --output datasets/ligand_atom_mappings/homology_25_10/test.jsonl

python scripts/build_ligand_atom_mappings.py \
    --manifest datasets/csv_files/ligand_25_10/test.csv \
    --output datasets/ligand_atom_mappings/ligand_25_10/test.jsonl
```
Each line is one pair: `ligand_id`, `tar_protein`/`tar_chain`, `src_protein`/`src_chain`,
atom counts, and `shared_atom_names` — the exact atoms `ligand_rmsd` is computed over. All
2,514 (homology) / 3,869 (ligand) evaluation pairs mapped successfully with no missing ligand
PDBs and no zero-overlap pairs. The per-ligand-id PDB files this script reads from live outside
the repo at `LIGAND_DIR` (`aligner_dl/utils/constants.py`); they are not redistributed here.

## Partition statistics & leakage check

Exact per-partition counts (pairs, unique protein chains, unique ligands, sequence clusters,
CATH superfamily groups) and train/test overlap on each of those axes:
```bash
python scripts/report_partition_stats.py \
    --cath_domain_list cath-classification-data/cath-domain-list.txt \
    --csv_dirs datasets/csv_files/homology_25_10 datasets/csv_files/ligand_25_10 \
    --out_dir results/partition_stats  # CATH version: see Version pins below
```
Writes `{split}_partition_counts.csv` and `{split}_leakage_check.csv` per split to `--out_dir`.

**Homology-safe split** (`datasets/csv_files/homology_25_10/`):

| Partition | Pairs | Unique chains | Unique ligands | Sequence clusters | Unique CATH groups |
|---|---|---|---|---|---|
| train | 79,527 | 10,120 | 697 | 7,255 | 2,098 |
| test | 2,514 | 1,354 | 138 | 968 | 581 |

**Ligand-grouped split** (`datasets/csv_files/ligand_25_10/`):

| Partition | Pairs | Unique chains | Unique ligands | Sequence clusters | Unique CATH groups |
|---|---|---|---|---|---|
| train | 97,731 | 10,338 | 675 | n/a (not clustered by sequence) | 2,061 |
| test | 3,869 | 1,757 | 119 | n/a | 734 |

**Leakage check, train vs. test:**

| Split | Chain overlap | Ligand overlap | Sequence-cluster overlap |
|---|---|---|---|
| `homology_25_10` | 0 / 1,354 test chains also in train | 121 / 138 test ligands also in train | 0 / 968 test clusters also in train |
| `ligand_25_10` | 454 / 1,757 test chains also in train | 0 / 119 test ligands also in train | n/a |

Each split's own leakage guarantee holds exactly as designed: the homology-safe split has zero
chain/cluster overlap between train and test (sequence identity is what it protects against); the
ligand-grouped split has zero ligand overlap (ligand identity is what it protects against). The
non-zero ligand overlap in the homology split and non-zero chain overlap in the ligand split are
expected, not a leak — a small-molecule ligand (e.g. ATP, ZN) legitimately recurs across unrelated
protein families, and a protein chain can legitimately appear paired with different ligands across
partitions when the split criterion is ligand identity rather than sequence identity.

Every chain in both splits matched an entry in the CATH domain list used above (10,120/10,120
train, 1,354/1,354 test for homology; 10,336/10,338, 1,757/1,757 for ligand).

## Version pins

- **CATH**: v4.4.0, file date 16 Dec 2024 (`cath-classification-data/cath-domain-list.txt`
  header). Download from the [CATH classification FTP](http://download.cathdb.info/cath/releases/).
  This was previously unstated anywhere in code or docs.
- **BioLiP**: non-redundant snapshot `BioLiP_nr_10082025.txt` (filename-embedded date, no
  separate upstream release string exists for BioLiP snapshots). Download from
  [BioLiP2](https://zhanggroup.org/BioLiP/). This was previously unstated anywhere in code
  or docs beyond the raw (gitignored) filename.

## Random seeds

Seed usage is scattered across the codebase rather than centralized; this is now
documented explicitly rather than fixed, to avoid changing training/eval behavior silently:

| Location | Value | Controls |
|---|---|---|
| `aligner_dl/trainers/lightning_trainer.py` `--seed` (CLI) | default 41 | `L.seed_everything` at training start |
| `aligner_dl/configs/loc_align.yaml` per-dataset `seed` | 41 (train), 81 (test) | dataset-level shuffling/sampling |
| `aligner_dl/models/layers/norm.py` | hardcoded `torch.manual_seed(42)` | **not** controlled by `--seed`; fixed regardless of CLI value |
| `aligner_dl/datasets/base_pair_dataset.py` `set_seed()` | see call sites | dataset construction |
| `aligner_dl/datasets/utils/split_dataset.py` | hardcoded `random.seed(42)` | MMseqs2 clustering / split assignment |
| `miners/parsers/biolip_reader.ipynb` | `random.seed(42)` / `np.random.seed(42)` (added by this change) | Stage B combination subsampling and probability-weighted sampling — **not seeded** in the run that produced the currently-released manifests; the manifests are released verbatim to sidestep this rather than claim retroactive reproducibility |

To reproduce training as closely as possible: use `--seed 41` (the default) and note that
`aligner_dl/models/layers/norm.py`'s hardcoded seed is independent of this flag.

## Training

```bash
conda env create -f inference_env.yaml   # or your own training env with the same deps
conda activate inference_env

# Homology-safe split (main model)
python -m aligner_dl.trainers.lightning_trainer \
    --config aligner_dl/configs/loc_align.yaml \
    --log_dir logs/loc_align \
    --seed 41 \
    --device gpu

# Ligand-grouped split (main model; the ablations use
# loc_align_ligand_split_quality0.yaml / loc_align_ligand_split_ligandloss0.yaml)
python -m aligner_dl.trainers.lightning_trainer \
    --config aligner_dl/configs/loc_align_ligand_split.yaml \
    --log_dir logs/loc_align_ligand_split \
    --seed 41 \
    --device gpu
```
`--validate_only` runs validation only against an existing checkpoint. Baseline/ablation
configs (`aligner_dl/configs/{softalign,tm_align,usalign,usalign_fns,dali,apoc,plasma}.yaml`)
follow the same CLI, pointed at the corresponding `test_baseline*.csv` manifest.

### Training protocol

The test set was never used for training. Hyperparameters were set using the training data only. No model was trained on `train.csv` + `test.csv`
combined. The Lightning "validation" loop in the code evaluates `test.csv` (the config key is
still `validation`); it was run once per epoch and logged (Comet metrics and
`per_sample_results_{epoch}.csv`).

- **Homology split** (`checkpoints/baseline/epoch=9-step=87120.ckpt`): trained on `train.csv`
  for the full 10-epoch schedule; the final checkpoint was used.
- **Ligand split** (`checkpoints/baseline-ligand-split/epoch=8-step=97191.ckpt`): the checkpoint
  after 9 of the 10 scheduled epochs was used (the learning-rate schedule spans 10 epochs). To reproduce, use
  `aligner_dl/configs/loc_align_ligand_split.yaml` (and `loc_align_ligand_split_quality0.yaml` /
  `loc_align_ligand_split_ligandloss0.yaml` for the two ablations). These set
  `trainer.max_epochs: 9` while keeping the OneCycleLR length at 10 epochs
  (`epochs: 10`, `steps_per_epoch: 10780` → `total_steps` 107,800, the value stored in the
  original checkpoints) and the corr-RMSD loss-weight ramp at 10 epochs
  (`trainer.schedule_epochs: 10`). Training therefore ends at global step 97,191
  (9 × 10,799 batches), matching the released checkpoints.

Mechanics, for reference:
- **Checkpointing.** `aligner_dl/trainers/lightning_trainer.py` registers one callback,
  `ModelCheckpoint(save_top_k=-1, every_n_epochs=1)`, which saves every epoch. It has no
  `monitor` and no `save_last`, and there is no `EarlyStopping` callback.
- **LR schedule.** `OneCycleLR` is stepped every optimizer step
  (`aligner_dl/models/soft_bb_base.py::configure_optimizers`). Its length comes only from the
  config's `optimizer.args.scheduler.args` (`epochs × steps_per_epoch`), never from
  `trainer.max_epochs`. That method also sets `'monitor': 'valid_loss'`, but Lightning only
  reads `monitor` for `ReduceLROnPlateau`, so it has no effect here. The `model_config.yaml`
  saved next to each checkpoint does not contain the scheduler block, because
  `SoftBBBase.__init__` pops it from the optimizer config; the schedule is in the training
  YAML.
- **Loss-weight ramp.** The corr-RMSD loss weight (`aligner_dl/losses/soft_bb_loss.py::update_lambda`)
  follows a fixed sine ramp over `global_step / total_steps`, where
  `total_steps = len(train_loader) × trainer.schedule_epochs` (default: `trainer.max_epochs`).

A CPU-only check (build the model from each ligand-split config, step its optimizer and
scheduler to step 97,191, and compare with the saved checkpoints) reproduces the checkpoints'
OneCycleLR `total_steps` (107,800), learning rate at step 97,191 (2.3988729944e-05) and
corr-RMSD loss weight (0.990151; 0 for `quality0`) exactly.

## Evaluation

```bash
python scripts/offline_metrics.py
```
Reads per-experiment result CSVs from `ABLATION_DIRS` (currently hardcoded absolute paths at
the top of the script — adjust to your environment) and reports success rate under
`SUCCESS_CRITERIA` (`corr_rmsd < 2.0`, `ligand_rmsd < 4.0`, `atom_type_fraction > 0.5`; baseline
aligners are judged on `ligand_rmsd` alone). `scripts/offline_metrics_apo.py` is the analogous
script for the apo/holo reanalysis (tracked separately, see that script's own splits under
`test_apo_apo.csv`/`test_apo_holo.csv`/`test_holo_holo_subset.csv`).
It reads only `holo_holo_subset.csv`, `apo_apo.csv` and `apo_holo.csv` of each directory in
`APO_REANALYSIS_SPLIT_DIRS`, each with `ligand_rmsd` in Angstrom (a column that is below 10 for
every pair is the loss-scale value and is rejected), and writes `all_experiments_with_success.csv`
and `success_rates.csv` next to them. Each split's tables must come from that split's own
checkpoint: the ligand-split directory holds the evaluation with the ligand-split checkpoint.

## Tests

Thin unit tests and a Table 1 reproducibility check live in `tests/`. Run them from the repo root
with the `localign_infer3` environment, which has every dependency the tests use:

```bash
python -m unittest discover -s tests -v
```

The reproducibility test reads `ablation_dfs/{homology,ligand}_split/baseline.csv` under DATA_ROOT,
and is skipped when those files are absent, for example in a fresh clone without the data. From a
fresh clone, point it at a checkout with the data: `LOCALIGN_DATA_ROOT=/path/to/LocAlign python -m
unittest discover -s tests`. The AUC tests are skipped when scikit-learn is not installed.

## Known limitations

- Only the homology-split and ligand-split main checkpoints are released; retrain the ablations
  from the documented commands to reproduce their numbers.
- Per-pair evaluation outputs depend on the GPU type. The same checkpoint and config give identical
  per-pair results on GPUs of one type (V100), but on another type (L40S) most per-pair values
  change while the aggregates agree. For example, the ligand-split binding-site fraction is 0.1615
  on both. The top-k correspondence selection turns small numeric differences into different
  correspondences. Pairs with a chain above `max_length` atoms are also randomly subsampled, so
  they depend on the seed.
- `miners/utils/process_pair.py`'s `min_ligand_atoms=3` parameter is declared but never applied
  in the function body — a dead filter, not a functioning exclusion criterion.
