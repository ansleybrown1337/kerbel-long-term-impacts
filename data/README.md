# Data

The repository currently keeps raw, processed, and model-ready tables together under `data/` and `out/`. The proposed Zenodo layout separates them without changing the active working tree:

- `data/raw/`: original monitoring and management inputs presently in `data/`;
- `data/processed/`: pipeline products presently in `out/pipeline_csvs/` and the active cleaned model table;
- `results/`: saved Bayes, ML, and comparison results, not input data.

See `docs/output_manifest.csv` for exact current-to-release paths and `data/data_dictionary.csv` for the release dictionary template.

## Units and analytical levels

- Concentration: mg/L unless a source column explicitly states otherwise.
- Runoff volume: L in row/event model tables; some publication volume tables use kL.
- Bayesian annual load draws: g.
- ML annual and row-reconstructed loads: mg.
- Comparison cumulative loads: kg.

The accepted upstream event key includes `Date + Year + Irrigation + Rep + Treatment + SampleID + MeasureMethod`. Before definitive release interpretation, document whether first-flush, outflow, and duplicate SampleIDs are separate physical load-bearing events or repeated measurements within one plot runoff event.

Do not edit raw source measurements in place unless a provenance-supported
transcription correction has been explicitly confirmed. Record confirmed source
corrections in `source_corrections_v3p0.csv`; record cleaning, alias mapping,
censoring, unit conversion, and aggregation rules in the processed-data
dictionary and workflow documentation.
