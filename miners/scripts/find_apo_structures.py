"""
Find genuine apo structures for the pairs in LocAlign's validation CSVs (reviewer
comment #1 reanalysis). See the approved plan at
/home/iscb/wolfson/hagairavid/.claude/plans/gleaming-greeting-stallman.md.

Usage:
    python miners/scripts/find_apo_structures.py --config miners/configs/apo_discovery_config.json
"""
import argparse
import json
import logging
import multiprocessing
import os
from datetime import datetime

import pandas as pd

from miners.apo_discovery.constants import DEFAULT_MIN_OVERLAP, DEFAULT_POCKET_DISTANCE_THRESH
from miners.apo_discovery.finder import ApoResult, find_apo_structure
from miners.objects.protein import Protein
from miners.utils.constants import LIGAND_DIR
from miners.utils.misc import save_results_to_csv

CHUNK_SIZE = 200
N_WORKERS = 20

start_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
os.makedirs(os.path.join("miners", "logs", "find_apo_structures"), exist_ok=True)
logging.basicConfig(
    filename=os.path.join("miners", "logs", "find_apo_structures", start_time + ".log"),
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)


def collect_unique_triples(df_paths: list[str]) -> list[tuple[str, str, str]]:
    """Unique (pdb_id, chain_id, ligand_id) triples across tar_*/src_* columns of all
    given CSVs, so shared chains (within or across splits) are only looked up once."""
    triples = set()
    for path in df_paths:
        df = pd.read_csv(path, dtype=str)
        for prefix in ("tar", "src"):
            cols = [f"{prefix}_protein", f"{prefix}_chain", "ligand_id"]
            for _, row in df[cols].dropna().iterrows():
                triples.add((row[f"{prefix}_protein"], row[f"{prefix}_chain"], row["ligand_id"]))
    return sorted(triples)


def _process_triple(args: tuple[str, str, str, dict]) -> dict:
    pdb_id, chain_id, ligand_id, cfg = args
    try:
        protein = Protein(pdb_name=pdb_id, chain_id=chain_id, ligand_name=ligand_id, ligand_dir=LIGAND_DIR, save_models=True)
        holo_chain = protein.get_model(0, only_chain=True)
        holo_ligand_residue = protein.get_ligand_residues()[0]
    except Exception as e:
        result = ApoResult(ref_pdb_id=pdb_id, ref_chain_id=chain_id, ligand_id=ligand_id, status="holo_init_failed")
        result.rejection_log.append(str(e))
        return result.__dict__

    try:
        result = find_apo_structure(
            ref_pdb_id=pdb_id,
            ref_chain_id=chain_id,
            ligand_id=ligand_id,
            holo_ligand_residue=holo_ligand_residue,
            holo_chain=holo_chain,
            structure_save_dir=cfg["structure_save_dir"],
            pdbe_cache_dir=cfg["pdbe_cache_dir"],
            structure_cache_dir=cfg["structure_dl_cache_dir"],
            min_overlap=cfg.get("min_overlap", DEFAULT_MIN_OVERLAP),
            distance_thresh=cfg.get("pocket_distance_thresh", DEFAULT_POCKET_DISTANCE_THRESH),
        )
    except Exception as e:
        # A batch job over thousands of real-world PDB entries WILL hit unforeseen
        # edge cases (as one already did - a multi-character candidate chain id
        # crashed the whole multiprocessing run and lost an in-flight chunk of
        # results). One bad triple must not take down the rest of the batch.
        result = ApoResult(ref_pdb_id=pdb_id, ref_chain_id=chain_id, ligand_id=ligand_id, status="unexpected_error")
        result.rejection_log.append(str(e))
    return result.__dict__


def run(
    triples: list[tuple[str, str, str]],
    cfg: dict,
    save_path: str,
    debug: bool = False,
    prev_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    args_list = [(pdb_id, chain_id, ligand_id, cfg) for pdb_id, chain_id, ligand_id in triples]
    result_list = []

    if not debug:
        pool = multiprocessing.Pool(N_WORKERS)
        for i in range(0, len(args_list), CHUNK_SIZE):
            chunk = args_list[i:i + CHUNK_SIZE]
            results_async = [pool.apply_async(_process_triple, (a,)) for a in chunk]
            result_list += [r.get() for r in results_async]
            save_results_to_csv(pd.DataFrame(result_list), start_time, "apo_lookup_temp_results", prev_df)
            logger.info(f"Processed {len(result_list)}/{len(args_list)} triples")
        pool.close()
        pool.join()
    else:
        result_list = [_process_triple(a) for a in args_list]

    df = save_results_to_csv(pd.DataFrame(result_list), start_time, "apo_lookup_results", prev_df, save_path=save_path)
    logger.info(f"Saved {len(df)} apo-lookup results to {save_path}")
    return df


def _row_key(row: pd.Series, prefix: str) -> tuple[str, str, str]:
    return (str(row[f"{prefix}_protein"]), str(row[f"{prefix}_chain"]), str(row["ligand_id"]))


def _resolve_side(found: bool, hit: dict | None, orig_protein: str, orig_chain: str) -> tuple:
    if found:
        return hit["synthetic_id"], hit["apo_chain_id"], "apo", hit["apo_pdb_id"], hit["apo_resolution"]
    return orig_protein, orig_chain, "holo", None, None


def build_apo_tables(df_paths: list[str], lookup_df: pd.DataFrame) -> dict[str, int]:
    """From the apo_lookup_results and each original val.csv, build the three new
    pair tables per the confirmed partition rule: holo-holo = superset (apo exists
    for >=1 side, untouched structures); apo-apo = apo exists for both sides;
    apo-holo = apo exists for exactly one side. Written alongside each source CSV
    as val_{holo_holo_subset,apo_apo,apo_holo}.csv.
    """
    lookup_index = {
        (r["ref_pdb_id"], r["ref_chain_id"], r["ligand_id"]): r
        for r in lookup_df.to_dict("records")
    }
    row_counts = {}

    for path in df_paths:
        df = pd.read_csv(path, dtype=str)
        holo_holo_rows, apo_apo_rows, apo_holo_rows = [], [], []

        for _, row in df.iterrows():
            tar_hit = lookup_index.get(_row_key(row, "tar"))
            src_hit = lookup_index.get(_row_key(row, "src"))
            tar_found = bool(tar_hit and tar_hit["status"] == "found")
            src_found = bool(src_hit and src_hit["status"] == "found")

            if not (tar_found or src_found):
                continue

            base = {"ligand_id": row["ligand_id"]}
            if "cath_degree" in row:
                base["cath_degree"] = row["cath_degree"]
            if "tar_ligand" in row:
                base["tar_ligand"] = row["tar_ligand"]
            if "src_ligand" in row:
                base["src_ligand"] = row["src_ligand"]
            # Required by aligner_dl/models/soft_bb_base.py:on_validation_epoch_end
            # (keys_to_keep includes these; a missing column is a hard KeyError, not
            # just missing metadata). The ligand itself is unchanged when a side is
            # swapped to apo - only the receiving backbone differs - so these carry
            # over verbatim from the original holo row.
            if "tar_ligand_n_atoms" in row:
                base["tar_ligand_n_atoms"] = row["tar_ligand_n_atoms"]
            if "src_ligand_n_atoms" in row:
                base["src_ligand_n_atoms"] = row["src_ligand_n_atoms"]

            holo_holo_rows.append({
                **base,
                "tar_protein": row["tar_protein"], "tar_chain": row["tar_chain"],
                "src_protein": row["src_protein"], "src_chain": row["src_chain"],
                "tar_protein_source": "holo", "src_protein_source": "holo",
            })

            tar_p, tar_c, tar_src, tar_apo_id, tar_apo_res = _resolve_side(
                tar_found, tar_hit, row["tar_protein"], row["tar_chain"]
            )
            src_p, src_c, src_src, src_apo_id, src_apo_res = _resolve_side(
                src_found, src_hit, row["src_protein"], row["src_chain"]
            )
            record = {
                **base,
                "tar_protein": tar_p, "tar_chain": tar_c, "tar_protein_source": tar_src,
                "tar_apo_pdb_id": tar_apo_id, "tar_apo_resolution": tar_apo_res,
                "src_protein": src_p, "src_chain": src_c, "src_protein_source": src_src,
                "src_apo_pdb_id": src_apo_id, "src_apo_resolution": src_apo_res,
            }
            if tar_found and src_found:
                apo_apo_rows.append(record)
            elif tar_found != src_found:
                apo_holo_rows.append(record)

        base_name = os.path.splitext(os.path.basename(path))[0]
        out_dir = os.path.dirname(path)
        for suffix, rows in [
            ("holo_holo_subset", holo_holo_rows),
            ("apo_apo", apo_apo_rows),
            ("apo_holo", apo_holo_rows),
        ]:
            out_path = os.path.join(out_dir, f"{base_name}_{suffix}.csv")
            pd.DataFrame(rows).to_csv(out_path, index=False)
            row_counts[out_path] = len(rows)
            logger.info(f"Wrote {len(rows)} rows to {out_path}")

    return row_counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Find genuine apo structures for LocAlign validation pairs")
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("-d", "--debug", action="store_true", help="Disable multiprocessing")
    parser.add_argument("-p", "--save_path", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    prev_df = None
    processed = set()
    if cfg.get("prev_lookup_path"):
        prev_df = pd.read_csv(cfg["prev_lookup_path"])
        processed = set(zip(prev_df["ref_pdb_id"], prev_df["ref_chain_id"], prev_df["ligand_id"]))

    triples = collect_unique_triples(cfg["df_paths"])
    triples = [t for t in triples if t not in processed]
    logger.info(f"{len(triples)} unique (pdb,chain,ligand) triples to process ({len(processed)} already done)")

    save_path = args.save_path or os.path.join(cfg["structure_save_dir"], "apo_lookup_results.csv")
    lookup_df = run(triples, cfg, save_path, debug=args.debug, prev_df=prev_df)

    build_apo_tables(cfg["df_paths"], lookup_df)


if __name__ == "__main__":
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()
