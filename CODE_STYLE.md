# Code style

## Comments

- Do not add comments inside code unless they are critical, such as a non-obvious constraint or a workaround that would otherwise be removed by mistake.
- Put explanations in the docstring of the function, method or class instead. State what the code does and any non-obvious behavior, units or scale there.
- Do not leave commented-out code, and do not narrate what the next line does.
- Linter directives (`# noqa`, `# type: ignore`) are fine.

## Python

- Use built-in generic types and `X | None`: `dict[str, int]`, `list[str]`, `str | None`, not `typing.Dict`, `List` or `Optional`.
- Put each argument of a function or method on its own line below the `def` line:

  ```python
  def fetch_smiles(
      ccd_id: str,
      timeout: float = 15.0,
  ) -> str | None:
  ```

- Put helper functions in a utils module (for example `aligner_dl/utils/`), or at the bottom of the script, below the main logic and above the `if __name__ == "__main__":` block.
- Do not use absolute paths, in code, configs, notebooks or data files. Define paths relative to the repo root in `aligner_dl/utils/constants.py` (`REPO_ROOT`, `DATA_ROOT` for untracked repo data such as `checkpoints/` and `ablation_dfs/`, `EXTERNAL_ROOT` for data and tool sources that live next to the repo) and import them. In YAML configs write `checkpoints/...` and `datasets/...` (resolved against `DATA_ROOT`) or `external/...` (resolved against `EXTERNAL_ROOT`); `resolve_config_paths` in `aligner_dl/utils/config_paths.py` resolves them. Find executables on `PATH` with `find_binary`; do not point to environment-specific interpreters or binaries. Set `LOCALIGN_DATA_ROOT` / `LOCALIGN_EXTERNAL_ROOT` to relocate the data, for example from a git worktree.
- Do not hard-code literal strings such as URLs, API field names or file paths inside functions. Define them in `aligner_dl/utils/constants.py` and import them.

## Results and generated files

- Do not commit result files (per-sample outputs, success-rate tables) by default. Commit the code, configs and run scripts needed to regenerate them, and document the commands in the PR description.
- Do not keep one-off generator scripts once the files they generate are committed. Commit the generated configs instead.
- Do not commit SLURM batch scripts (`*.sbatch`, `*.slurm`). Document the submit command in the PR description instead.
