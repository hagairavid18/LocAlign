"""Ligand symmetry from the PDB Chemical Component Dictionary (CCD).

For each CCD code, fetch the ligand's SMILES from the RCSB Chemical Component API and count, with
RDKit, the atom permutations (graph automorphisms) that map the molecule onto itself. A count above
1 means the ligand has chemically equivalent atoms, so an RMSD computed by atom name can be too high
for a correctly placed ligand.

The counts are cached as JSON ({ccd_code: n_automorphisms}, null when the fetch or parse failed).
scripts/offline_metrics.py calls load_or_build_symmetry_counts(), which builds the cache when it is
missing. Building needs network access and RDKit.
"""
import json
import os
import sys
import time

import requests
from rdkit import Chem

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "aligner_dl"))
from utils.constants import (  # noqa: E402
    RCSB_CHEMCOMP_URL,
    RCSB_DESCRIPTOR_KEY,
    RCSB_SMILES_KEYS,
    RCSB_SMILES_STEREO_KEYS,
)


def fetch_smiles(
    ccd_id: str,
    session: requests.Session,
    prefer_stereo: bool = True,
    timeout: float = 15.0,
    attempts: int = 3,
    retry_wait: float = 2.0,
) -> str | None:
    """Fetch a SMILES string for a CCD code from the RCSB Chemical Component API.

    The stereo SMILES is preferred when prefer_stereo is set. The descriptor keys are tried in lower
    and upper case, since the API casing differs by version. A failed request is retried up to
    attempts times, retry_wait seconds apart, so that a transient error is not cached as a failure.
    """
    url = RCSB_CHEMCOMP_URL.format(ccd_id=ccd_id.upper())
    for attempt in range(1, attempts + 1):
        try:
            resp = session.get(url, timeout=timeout)
            resp.raise_for_status()
            break
        except requests.RequestException as e:
            print(f"  [warn] fetch attempt {attempt}/{attempts} failed for {ccd_id}: {e}", file=sys.stderr)
            if attempt == attempts:
                return None
            time.sleep(retry_wait)

    desc = resp.json().get(RCSB_DESCRIPTOR_KEY, {})
    keys = (RCSB_SMILES_STEREO_KEYS + RCSB_SMILES_KEYS) if prefer_stereo else RCSB_SMILES_KEYS
    for key in keys:
        if desc.get(key):
            return desc[key]
    return None


def count_automorphisms(
    smiles: str,
    include_hs: bool = False,
    use_chirality: bool = True,
) -> int | None:
    """Count the self-mapping atom permutations of a molecule (its automorphism group size).

    This is the number of index relabelings that leave the molecule graph, and with use_chirality
    its stereochemistry, unchanged. include_hs=False restricts it to heavy atoms, matching an RMSD
    computed on heavy-atom coordinates.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    if include_hs:
        mol = Chem.AddHs(mol)
    return len(mol.GetSubstructMatches(mol, uniquify=False, useChirality=use_chirality))


def build_symmetry_dict(
    ccd_codes,
    include_hs: bool = False,
    use_chirality: bool = True,
    sleep_between_requests: float = 0.1,
) -> dict[str, int | None]:
    """Fetch the SMILES of each CCD code and compute its automorphism count.

    Codes that fail to fetch or to parse map to None, with a warning on stderr. Requests are spaced
    by sleep_between_requests seconds to limit the load on the API.
    """
    results: dict[str, int | None] = {}
    session = requests.Session()
    session.headers.update({"Accept": "application/json"})

    for ccd_id in ccd_codes:
        print(f"Processing {ccd_id} ...")
        smiles = fetch_smiles(ccd_id, session)
        if smiles is None:
            print(f"  [warn] no SMILES found for {ccd_id}", file=sys.stderr)
            results[ccd_id] = None
        else:
            n = count_automorphisms(smiles, include_hs=include_hs, use_chirality=use_chirality)
            if n is None:
                print(f"  [warn] RDKit could not parse SMILES for {ccd_id}: {smiles}", file=sys.stderr)
            results[ccd_id] = n
        time.sleep(sleep_between_requests)

    return results


def load_or_build_symmetry_counts(
    ccd_codes,
    path: str,
) -> dict[str, int | None]:
    """Automorphism counts for ccd_codes, read from the JSON cache at path.

    Codes missing from the cache are fetched and the cache is rewritten, so the first run builds it.
    """
    counts: dict[str, int | None] = {}
    if os.path.exists(path):
        with open(path) as f:
            counts = json.load(f)
    missing = sorted(set(ccd_codes) - set(counts))
    if missing:
        counts.update(build_symmetry_dict(missing, include_hs=False, use_chirality=True))
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            json.dump(dict(sorted(counts.items())), f, indent=2)
        print(f"Saved {len(counts)} codes to {path}")
    return counts
