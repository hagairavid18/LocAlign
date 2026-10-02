# Code style

## Comments

- Do not add comments inside code unless they are critical, such as a non-obvious constraint or a workaround that would otherwise be removed by mistake.
- Put explanations in the docstring of the function, method or class instead. State what the code does and any non-obvious behavior, units or scale there.
- Do not leave commented-out code, and do not narrate what the next line does.
- Linter directives (`# noqa`, `# type: ignore`) are fine.

## Results and generated files

- Do not commit result files (per-sample outputs, success-rate tables) by default. Commit the code, configs and run scripts needed to regenerate them, and document the commands in the PR description.
- Do not keep one-off generator scripts once the files they generate are committed. Commit the generated configs instead.
- Do not commit SLURM batch scripts (`*.sbatch`, `*.slurm`). Document the submit command in the PR description instead.
