"""
Run full validation for a trained LocAlign checkpoint and report the binding-site
correspondence fraction metric (Reviewer 1, comment 3), alongside the existing
ligand RMSD / atom-type metrics for context.

Usage:
    python scripts/eval_binding_site_metric.py \
        --checkpoint_dir /path/to/checkpoints/baseline-ligand-split \
        --checkpoint_file "epoch=8-step=97191-v1.ckpt" \
        --output_dir results/binding_site_eval \
        --tag main_model
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "aligner_dl")))

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from datasets import ScanNetDataset
from models.utils.collate import custom_collate_fn
from models.utils.misc import build_object


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate binding-site correspondence metric on a checkpoint.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--checkpoint_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True, help="Short name for this model variant, used in output filenames.")
    parser.add_argument("--n_samples", type=int, default=None, help="Optional cap for a quick smoke run.")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--num_threads", type=int, default=8, help="torch intra-op CPU threads.")
    parser.add_argument("--data_root", type=str, default="/home/iscb/wolfson/hagairavid/LocAlign",
                         help="Base dir to resolve a relative df_path from dataset_config.yaml against "
                              "(data CSVs are gitignored and may not exist under a worktree checkout).")
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "cpu"],
                         help="Defaults to cuda if available, else cpu.")
    return parser.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.num_threads)
    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.checkpoint_dir, "model_config.yaml")) as f:
        model_cfg = yaml.safe_load(f)
    with open(os.path.join(args.checkpoint_dir, "dataset_config.yaml")) as f:
        dataset_cfg = yaml.safe_load(f)

    if args.n_samples is not None:
        dataset_cfg["args"]["n_samples"] = args.n_samples

    df_path = dataset_cfg["args"]["df_path"]
    if not os.path.isabs(df_path):
        dataset_cfg["args"]["df_path"] = os.path.join(args.data_root, df_path)

    print(f"[{args.tag}] Building dataset from {dataset_cfg['args']['df_path']} ...", flush=True)
    t0 = time.time()
    dataset = ScanNetDataset(**dataset_cfg["args"])
    print(f"[{args.tag}] Dataset built in {time.time()-t0:.1f}s, {len(dataset)} pairs.", flush=True)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=custom_collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
    )

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{args.tag}] Building model and loading checkpoint {args.checkpoint_file} onto {device} ...", flush=True)
    model = build_object(model_cfg, "models")
    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_file)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    assert not missing and not unexpected, f"State dict mismatch: missing={missing}, unexpected={unexpected}"
    model.to(device)
    model.eval()

    # The checkpoint's saved model_config.yaml predates the BindingSiteCorrespondence metric
    # (it's a training-time snapshot), so it won't be built by build_object above. Attach it
    # directly rather than editing the checkpoint's on-disk config.
    if "binding_site" not in model._metrics.metrics:
        from metrics import BindingSiteCorrespondence
        model._metrics.metrics["binding_site"] = BindingSiteCorrespondence()
        print(f"[{args.tag}] Attached BindingSiteCorrespondence metric (not present in checkpoint's saved config).", flush=True)

    print(f"[{args.tag}] Running validation over {len(dataset)} pairs, batch_size={args.batch_size} ...", flush=True)
    t_start = time.time()
    n_done = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            model.validation_step(batch)
            n_done += len(batch["metadata"])
            if batch_idx % 20 == 0:
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta_min = (len(dataset) - n_done) / rate / 60 if rate > 0 else float("nan")
                print(f"[{args.tag}] {n_done}/{len(dataset)} done, {elapsed:.0f}s elapsed, "
                      f"{rate:.2f} samples/s, ETA {eta_min:.1f} min", flush=True)

    total_time = time.time() - t_start
    print(f"[{args.tag}] Validation loop finished in {total_time:.0f}s for {n_done} samples.", flush=True)

    metrics = model._metrics.compute()

    summary = {
        "tag": args.tag,
        "checkpoint_dir": args.checkpoint_dir,
        "checkpoint_file": args.checkpoint_file,
        "n_samples": n_done,
        "total_time_seconds": total_time,
        "weighted_pocket_fraction_overall": metrics["weighted_pocket_fraction_overall"],
        "weighted_pocket_fraction_per_degree": metrics["weighted_pocket_fraction_per_degree"],
        "weighted_same_type_overall": metrics.get("weighted_same_type_overall"),
        "weighted_same_type_per_degree": metrics.get("weighted_same_type_per_degree"),
        "ligand_rmsd_below_4_total_proportion": metrics.get("ligand_rmsd_below_4_total_proportion"),
        "ligand_rmsd_per_degree": metrics.get("ligand_rmsd"),
        "counts_per_degree": metrics.get("counts_per_degree"),
        "total_count": metrics.get("total_count"),
    }

    summary_path = os.path.join(args.output_dir, f"{args.tag}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[{args.tag}] Wrote summary to {summary_path}", flush=True)

    # Per-sample CSV for downstream analysis.
    # Note: metrics['ligand_rmsd_per_sample'] (from PocketRMSD) is NOT used here — for
    # batch_size > 1 it holds the whole per-batch tensor per entry instead of a per-sample
    # scalar (a pre-existing indexing bug in pocket_ligand_rmsd.py, unrelated to this script).
    per_sample_rows = [
        {"cath_degree": cath_deg, "pocket_fraction": pocket_frac}
        for pocket_frac, cath_deg in zip(
            metrics["pocket_fraction_per_sample"], metrics["cath_degree_per_sample"]
        )
    ]
    per_sample_df = pd.DataFrame(per_sample_rows)
    per_sample_path = os.path.join(args.output_dir, f"{args.tag}_per_sample.csv")
    per_sample_df.to_csv(per_sample_path, index=False)
    print(f"[{args.tag}] Wrote per-sample results to {per_sample_path}", flush=True)

    print(f"[{args.tag}] weighted_pocket_fraction_overall = {summary['weighted_pocket_fraction_overall']:.4f}", flush=True)
    print(f"[{args.tag}] DONE.", flush=True)


if __name__ == "__main__":
    main()
