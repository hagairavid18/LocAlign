# Reproducibility

This document describes the exact data, splits, and commands needed to reproduce LocAlign's
training data, evaluation numbers, and (for the data pipeline) the pair manifests themselves.
It responds to a reviewer request to release: pair manifests, split labels, ligand atom
mappings, exclusion logs, the CATH version used, software commands, random seeds, and
evaluation scripts.

## What is released in this repo

| Artifact | Path |
|---|---|
| Homology-safe split (main model) | `datasets/csv_files/homology_25_10/{train,val}.csv` |
| Ligand-grouped split | `datasets/csv_files/ligand_25_10/{train,val}.csv` |
| Baseline-comparison manifests (same pairs + precomputed baseline results) | `datasets/csv_files/{homology,ligand}_25_10/val_baseline*.csv` |
| Ligand atom-name correspondence (evaluation pairs) | `datasets/ligand_atom_mappings/{homology,ligand}_25_10/val.jsonl` |
| Data pipeline scripts | `miners/parsers/biolip_reader.ipynb`, `miners/scripts/align.py`, `aligner_dl/datasets/utils/split_dataset.py` |
| Ligand-mapping generation script | `scripts/build_ligand_atom_mappings.py` |
| Evaluation scripts | `scripts/offline_metrics.py`, `scripts/offline_metrics_apo.py` |
| Training entrypoint | `aligner_dl/trainers/lightning_trainer.py` |

**Not released**: trained model checkpoints (11GB, no release infrastructure set up for this);
raw PDB/mmCIF structures, the raw BioLiP text dump, and the full CATH classification files
(all large, third-party, publicly redownloadable — see version pins below instead of
redistributing them); the `train.csv` splits' row-level ligand atom mappings (~80k-100k pairs —
these are recomputed on the fly by the training dataloader itself, see below, so materializing
them here would just be a slow, redundant copy of what running the code already does).
`train.csv` and `val.csv`/`val_baseline*.csv` manifests themselves *are* released in full.

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

**Stage D — train/val/test split** (`aligner_dl/datasets/utils/split_dataset.py::split_csv`):
```bash
# Homology-safe split (main model, aligner_dl/configs/loc_align.yaml)
python -m aligner_dl.datasets.utils.split_dataset \
    --input_csv calculated_cath/<N>_25_10_2025.csv \
    --output_dir datasets/csv_files/homology_25_10 \
    --test_size 0.2 --val_size 0.1 --mmseq_id_threshold 0.5

# Ligand-grouped split (no ligand shared across train/val/test)
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
`test.csv` in both directories is effectively empty — reported numbers use `val*.csv` as the
held-out evaluation set, not a separate test split.

## Exclusion logs

Historically, every filter above only printed a before/after row count — no filtered-out rows
were persisted. This PR makes that persistent going forward:
- `miners/parsers/biolip_reader.ipynb` writes each Stage B filter's dropped rows to
  `calculated_cath/exclusion_logs/{01_singleton_ligand,02_ligand_type,03_heavy_atom_count,04_cath_degree}_excluded.csv`.
- `aligner_dl/datasets/utils/split_dataset.py` writes Stage D's dropped rows (duplicates +
  Stage C failures) to `<output_dir>/excluded_pairs.csv`.

We did not rerun the full historical pipeline to reconstruct the exact exclusion list behind
the already-committed `train`/`val` manifests (this would require re-touching the raw
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
    --manifest datasets/csv_files/homology_25_10/val.csv \
    --output datasets/ligand_atom_mappings/homology_25_10/val.jsonl

python scripts/build_ligand_atom_mappings.py \
    --manifest datasets/csv_files/ligand_25_10/val.csv \
    --output datasets/ligand_atom_mappings/ligand_25_10/val.jsonl
```
Each line is one pair: `ligand_id`, `tar_protein`/`tar_chain`, `src_protein`/`src_chain`,
atom counts, and `shared_atom_names` — the exact atoms `ligand_rmsd` is computed over. All
2,514 (homology) / 3,869 (ligand) evaluation pairs mapped successfully with no missing ligand
PDBs and no zero-overlap pairs. The per-ligand-id PDB files this script reads from live outside
the repo at `LIGAND_DIR` (`aligner_dl/utils/constants.py`); they are not redistributed here.

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
| `aligner_dl/configs/loc_align.yaml` per-dataset `seed` | 41 (train), 81 (val) | dataset-level shuffling/sampling |
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

python -m aligner_dl.trainers.lightning_trainer \
    --config aligner_dl/configs/loc_align.yaml \
    --log_dir logs/loc_align \
    --seed 41 \
    --device gpu
```
`--validate_only` runs validation only against an existing checkpoint. Baseline/ablation
configs (`aligner_dl/configs/{softalign,tm_align,usalign,usalign_fns,dali,apoc,plasma}.yaml`)
follow the same CLI, pointed at the corresponding `val_baseline*.csv` manifest.

## Evaluation

```bash
python scripts/offline_metrics.py
```
Reads per-experiment result CSVs from `ABLATION_DIRS` (currently hardcoded absolute paths at
the top of the script — adjust to your environment) and reports success rate under
`SUCCESS_CRITERIA` (`corr_rmsd < 2.0`, `ligand_rmsd < 4.0`, `atom_type_fraction > 0.5`; baseline
aligners are judged on `ligand_rmsd` alone). `scripts/offline_metrics_apo.py` is the analogous
script for the apo/holo reanalysis (tracked separately, see that script's own splits under
`val_apo_apo.csv`/`val_apo_holo.csv`/`val_holo_holo_subset.csv`).

## Known limitations

- Checkpoints are not released; retrain from the documented commands to reproduce reported
  numbers.
- `scripts/offline_metrics.py`'s `ABLATION_DIRS` are hardcoded absolute paths from the original
  development machine.
- `miners/utils/process_pair.py`'s `min_ligand_atoms=3` parameter is declared but never applied
  in the function body — a dead filter, not a functioning exclusion criterion.
