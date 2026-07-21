# Legacy artifact migration for v3p0

Migration completed on 2026-07-19 after explicit user approval. No scientific
artifact was deleted; files and directories were moved within the repository.
Only empty source directories were removed afterward.

## Routing

- Loose Bayesian v2p1 figures: `figs/` to `figures/bayes/v2p1/`.
- Historical Bayesian figures: `figs/old_bayes_figs/` to the version parsed
  from each filename under `figures/bayes/<version>/`.
- Historical Bayesian tables: `out/old bayes output/` and loose v2p1 CSVs under
  `out/` to `results/bayes/<version>/`.
- Legacy Bayesian fits and chain CSVs: `code/out_cmdstanr/` to
  `results/bayes/<version>/cmdstan/`; dashboard images moved to the matching
  figure folders.
- Versioned Bayes-versus-ML directories: routed to
  `figures/comparison/<version>/` and `results/comparison/<version>/`.
- Unversioned comparison checks: routed to `comparison/unversioned/`.
- The unversioned CatBoost run: routed to `ml/unversioned/`.
- General reference images (`banner`, DAG, raw TSS, and STIR comparison): routed
  to `figures/reference/`.

The migration moved 148 individually classified files, 29 legacy result/figure
directories, and 23 CmdStan artifacts totaling 23.55 GiB. Destination collision
checks ran before each move. Afterward, `out/` retained only pipeline products:
`wq_cleaned.csv`, `pipeline_csvs/`, and the STIR workbook.
