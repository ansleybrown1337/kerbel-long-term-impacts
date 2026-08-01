# Accepted v3p0 physical-event archive

This folder preserves the accepted v3p0 source and documentation snapshot before the v3p1 event-key and mean-per-plot correction. It is historical code and is not an active workflow.

## Source snapshot

- `code/bayes/`: v3p0 Stan, production R, synchronized R Markdown, and the compiled v3p0 executable that was removed from active source.
- `code/ml/`: v3p0 training, saved-model reconstruction, and plotting code.
- `code/comparison/`: v3p0 comparison and publication-export code.
- `code/shared/`: the v3p0 shared physical-event contract and preflight.
- `config/physical_event_v3p0.json`: accepted v3p0 configuration.
- `tests/test_physical_event_v3p0.py`: accepted focused tests.
- `docs/`: accepted v3p0 method, workflow, and reproducibility documentation copied before v3p1 edits.

## Archived output locations

- `results/bayes/old_versions/v3p0_physical_event/`
- `results/ml/old_versions/v3p0_physical_event/`
- `results/comparison/old_versions/v3p0_physical_event/`
- `figures/bayes/old_versions/v3p0_physical_event/`
- `figures/ml/old_versions/v3p0_physical_event/`
- `figures/comparison/old_versions/v3p0_physical_event/`
- `validation/preflight/old_versions/physical_event_v3p0/`

The archive was made by copying source/document files before modifying active code and by moving v3p0 output directories without regenerating their contents. No accepted v3p0 model was rerun.

Large draw ledgers and CmdStan chain CSVs remain physically present in the
archived output folders but continue to match the repository's existing ignore
rules. They are local runtime artifacts and will not be added by an ordinary
`git add -A`.
