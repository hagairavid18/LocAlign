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
- Do not use absolute paths, in code, configs, notebooks or data files. Tracked paths (`datasets/`, `example_inputs/`, `aligner_dl/configs/`, ...) hang off `CHECKOUT_ROOT`, the root of the code's own checkout; untracked data (`checkpoints/`, `ablation_dfs/`, `results/`) hangs off `DATA_ROOT`, which falls back to the main checkout in a git worktree. Both are in `aligner_dl/utils/constants.py`. Data and tools outside the repo each have their own constant there (`SCANNET_DIR`, `LIGAND_DIR`, `SOFTALIGN_DIR`, ...) with a `LOCALIGN_*` environment variable override, defaulting to a path relative to `DATA_ROOT`; import them. In YAML configs write relative paths (`df_path` resolves against `CHECKOUT_ROOT`, `ckpt_path` against `DATA_ROOT`) or `${SCANNET_DIR}`-style placeholders; `resolve_config_paths` in `aligner_dl/utils/config_paths.py` resolves them. Find executables on `PATH` with `find_binary`. Machine-specific values go in the untracked `.env` in the repo root (copy `.env.example`), which `constants.py` loads automatically; precedence is shell/job environment, then `.env`, then the default in code. Set `LOCALIGN_DATA_ROOT` in the real environment to relocate the data. Scripts that need the repo root for `sys.path` derive it from their own `__file__`.
- Do not hard-code literal strings such as URLs, API field names or file paths inside functions. Define them in `aligner_dl/utils/constants.py` and import them.

## Results and generated files

- Do not commit result files (per-sample outputs, success-rate tables) by default. Commit the code, configs and run scripts needed to regenerate them, and document the commands in the PR description.
- Do not keep one-off generator scripts once the files they generate are committed. Commit the generated configs instead.
- Do not commit SLURM batch scripts (`*.sbatch`, `*.slurm`). Document the submit command in the PR description instead.
