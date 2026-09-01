# Data

This directory contains the source monitoring and management tables used by the
reproducible pipeline. The pipeline writes intermediate audit tables under
`out/pipeline_csvs/` and the model-ready table to `out/wq_cleaned.csv`.

Key source tables include the master water-quality record, tillage operations,
crop records, residue observations and dates, measurement-method metadata,
furrow tire-compaction records, and provenance-supported source corrections.
Column definitions and units are documented in `data_dictionary.csv` and
`docs/methods/data_unit_dictionary.md`.

The physical-event key is `Year + Irrigation + Rep + Treatment`. Date, sample
identifier, analyte, and measurement method are observation metadata rather
than event-identity fields.

Do not edit raw source measurements in place unless a provenance-supported
transcription correction has been explicitly confirmed. Record such changes in
`source_corrections.csv`; record cleaning, alias mapping, censoring, unit
conversion, and aggregation rules in the data dictionary and workflow docs.
