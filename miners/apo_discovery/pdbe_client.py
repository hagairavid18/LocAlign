import json
import logging
import os
import uuid

import requests

from miners.apo_discovery.constants import PDBE_BEST_STRUCTURES_URL, PDBE_UNIPROT_MAPPING_URL

logger = logging.getLogger(__name__)


def fetch_json(url: str, cache_path: str, timeout: int = 30) -> dict | None:
    """Single-attempt GET + JSON-file cache. Returns None on any failure
    (matches the single-attempt-no-retry philosophy of Protein._init_structure).

    Cache writes go through a per-call temp path + atomic os.replace, NOT a direct
    write to cache_path - many parallel workers can request the SAME url (e.g. two
    different reference chains mapping to the same UniProt accession), and a direct
    write lets concurrent writers interleave into a corrupted-but-still-parseable
    file. Atomic rename means a reader only ever sees no file or a complete one.
    """
    if os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cached JSON {cache_path}: {e}")

    try:
        response = requests.get(url, timeout=timeout)
        if response.status_code != 200:
            logger.info(f"GET {url} returned status {response.status_code}")
            return None
        data = response.json()
    except Exception as e:
        logger.warning(f"Failed to fetch {url}: {e}")
        return None

    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = f"{cache_path}.tmp_{os.getpid()}_{uuid.uuid4().hex[:8]}"
        with open(tmp_path, "w") as f:
            json.dump(data, f)
        os.replace(tmp_path, cache_path)  # atomic on the same filesystem
    except Exception as e:
        logger.warning(f"Failed to cache JSON to {cache_path}: {e}")

    return data


def get_uniprot_accessions_for_chain(pdb_id: str, chain_id: str, cache_dir: str) -> list[dict]:
    """Map a (pdb_id, author chain_id) to the UniProt accession(s) it covers, with the
    reference chain's own UniProt residue range (merged across segments if multiple).

    Returns: [{"accession": str, "unp_start": int, "unp_end": int}, ...]
    """
    pdb_id_lower = pdb_id.lower()
    cache_path = os.path.join(cache_dir, "uniprot_mappings", f"{pdb_id_lower}.json")
    data = fetch_json(PDBE_UNIPROT_MAPPING_URL.format(pdb_id=pdb_id_lower), cache_path)
    if not data:
        return []

    entry = data.get(pdb_id_lower, {})
    uniprot_section = entry.get("UniProt", {})

    results = []
    for accession, accession_data in uniprot_section.items():
        segments = [
            m for m in accession_data.get("mappings", [])
            if m.get("chain_id") == chain_id
        ]
        if not segments:
            continue
        unp_start = min(m["unp_start"] for m in segments)
        unp_end = max(m["unp_end"] for m in segments)
        results.append({"accession": accession, "unp_start": unp_start, "unp_end": unp_end})

    return results


def get_best_structures(accession: str, cache_dir: str) -> list[dict]:
    """All candidate PDB entries for a UniProt accession, with resolution + UniProt range.

    Returns the raw list of dicts from the PDBe API, each containing at least:
    pdb_id, chain_id, resolution, unp_start, unp_end, coverage, experimental_method, tax_id.
    """
    cache_path = os.path.join(cache_dir, "best_structures", f"{accession}.json")
    data = fetch_json(PDBE_BEST_STRUCTURES_URL.format(accession=accession), cache_path)
    if not data:
        return []
    return data.get(accession, [])
