import os
import shutil
import subprocess
from pathlib import Path

DATA_ROOT_ENV = 'LOCALIGN_DATA_ROOT'
EXTERNAL_ROOT_ENV = 'LOCALIGN_EXTERNAL_ROOT'
DATA_ROOT_SENTINEL = 'ablation_dfs'


def find_repo_root() -> Path:
    """Root of the repository containing this file."""
    return Path(__file__).resolve().parents[2]


def find_data_root(repo_root: Path) -> Path:
    """Directory that holds the untracked repo data (datasets, checkpoints, ablation_dfs, results).

    Resolution order: the LOCALIGN_DATA_ROOT environment variable; the repo root if it
    contains DATA_ROOT_SENTINEL; the main checkout of the repository (through the git
    common dir) when running from a worktree and it contains DATA_ROOT_SENTINEL; the
    repo root otherwise (a fresh clone without the untracked data).
    """
    env = os.environ.get(DATA_ROOT_ENV)
    if env:
        root = Path(env).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f'{DATA_ROOT_ENV}={env} is not a directory')
        return root
    if (repo_root / DATA_ROOT_SENTINEL).is_dir():
        return repo_root
    main_checkout = find_main_checkout(repo_root)
    if main_checkout is not None and (main_checkout / DATA_ROOT_SENTINEL).is_dir():
        return main_checkout
    return repo_root


def find_main_checkout(repo_root: Path) -> Path | None:
    """Main checkout of the repository when `repo_root` is a git worktree, else None."""
    git = shutil.which('git')
    if git is None:
        return None
    result = subprocess.run(
        [git, 'rev-parse', '--git-common-dir'],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    main_checkout = (repo_root / result.stdout.strip()).resolve().parent
    return main_checkout if main_checkout != repo_root else None


def find_external_root(data_root: Path) -> Path:
    """Directory holding data and tool sources that live next to the repo (scannet_2212, ligands, SoftAlign, ...).

    The LOCALIGN_EXTERNAL_ROOT environment variable, else the parent of the data root.
    """
    env = os.environ.get(EXTERNAL_ROOT_ENV)
    if env:
        return Path(env).expanduser().resolve()
    return data_root.parent


def find_binary(name: str) -> str:
    """Absolute path of the executable `name` found on PATH; raises FileNotFoundError if missing."""
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f'{name} was not found on PATH; install it or add it to PATH')
    return path
