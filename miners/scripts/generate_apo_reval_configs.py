"""
Generate the 6 Lightning validation configs (3 apo/holo tables x 2 splits) needed to
regenerate LocAlign's per-sample metrics on the reviewer-comment-#1 reanalysis tables,
by copying the real baseline checkpoint's saved model/dataset config and swapping in
each new table's df_path. See the approved plan at
/home/iscb/wolfson/hagairavid/.claude/plans/gleaming-greeting-stallman.md, section 7.

Usage:
    python miners/scripts/generate_apo_reval_configs.py

Note: `trainer.validate_only: true` in the generated YAML is what actually gates
validate-vs-train in aligner_dl/trainers/lightning_trainer.py (the --validate_only CLI
flag exists but is unused) - so run each generated config as:
    python aligner_dl/trainers/lightning_trainer.py --config <generated_config>.yaml
"""
import os

import yaml

REPO_ROOT = "/home/iscb/wolfson/hagairavid/LocAlign"
# Real checkpoint behind the renamed checkpoints/baseline/ dir (confirmed byte-identical).
CKPT_PATH = os.path.join(
    REPO_ROOT,
    "checkpoints/baseline-embed01-ligand5-corr1-recycle4-radius05-sched-corr2/epoch=9-step=87120.ckpt",
)
MODEL_CONFIG_PATH = os.path.join(REPO_ROOT, "checkpoints/baseline/model_config.yaml")
DATASET_CONFIG_PATH = os.path.join(REPO_ROOT, "checkpoints/baseline/dataset_config.yaml")
CONFIG_OUT_DIR = os.path.join(REPO_ROOT, "aligner_dl/configs")

# checkpoints/baseline/dataset_config.yaml's saved base_scannet_path
# ("scannet_atom_types") is a stale/unused snapshot with zero coverage of the real
# validation proteins - confirmed 0/1354 homology + 0/1757 ligand real
# protein+chain tokens found there. scannet_2212 is what
# aligner_dl/configs/loc_align.yaml (the current master config) actually uses, and
# has 100% coverage of both val.csv splits, so it - not the checkpoint's saved
# value - is what both Phase B (materialize_apo_artifacts.py) and this re-eval
# config must point at.
BASE_SCANNET_PATH = "/home/iscb/wolfson/hagairavid/scannet_2212"

SPLITS = {
    "homology": os.path.join(REPO_ROOT, "datasets/csv_files/homology_25_10"),
    "ligand": os.path.join(REPO_ROOT, "datasets/csv_files/ligand_25_10"),
}
TABLES = ["holo_holo_subset", "apo_apo", "apo_holo"]


def main() -> None:
    with open(MODEL_CONFIG_PATH) as f:
        model_config = yaml.safe_load(f)
    with open(DATASET_CONFIG_PATH) as f:
        dataset_config = yaml.safe_load(f)

    os.makedirs(CONFIG_OUT_DIR, exist_ok=True)
    written = []

    for split_name, split_dir in SPLITS.items():
        for table_name in TABLES:
            df_path = os.path.join(split_dir, f"val_{table_name}.csv")
            exp_name = f"apo-reval-{split_name}-{table_name.replace('_', '-')}"

            validation_args = dict(dataset_config["args"])
            validation_args["df_path"] = df_path
            validation_args["base_scannet_path"] = BASE_SCANNET_PATH

            config = {
                "trainer": {
                    "exp_name": exp_name,
                    "tags": ["apo-reanalysis", split_name, table_name],
                    "max_epochs": 10,
                    "validate_only": True,
                    "check_val_every_n_epoch": 1,
                    "gradient_clipping": 30.0,
                    "ckpt_path": CKPT_PATH,
                },
                "model": model_config,
                "dataset": {
                    "validation": {
                        "name": dataset_config["name"],
                        "args": validation_args,
                    }
                },
                "dataloader": {
                    "train_batch_size": 8,
                    "valid_batch_size": 8,
                    "n_workers": 12,
                },
            }

            out_path = os.path.join(CONFIG_OUT_DIR, f"apo_reval_{split_name}_{table_name}.yaml")
            with open(out_path, "w") as f:
                yaml.dump(config, f, sort_keys=False)
            written.append(out_path)
            print(f"Wrote {out_path} (exp_name={exp_name}, df_path={df_path})")

    print(f"\n{len(written)} configs written. Run each with:")
    print("  <aligner_dl2 env python> aligner_dl/trainers/lightning_trainer.py --config <path>")


if __name__ == "__main__":
    main()
