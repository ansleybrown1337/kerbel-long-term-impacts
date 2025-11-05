# Running the data pipeline

### Quick-start examples

✅ **How to use:**

1. Open the **Anaconda Prompt**.
2. `cd` into your repo folder, e.g.

   ```bash
   cd "C:\Users\ansle\OneDrive\Documents\GitHub\kerbel-long-term-impacts"
   ```
3. Paste and run the command:

```bash
python code/main.py --wq-long out/kerbel_master_concentrations_long.csv --stir-events out/stir_events_long.csv --crops "data/crop records.csv" --records data/tillage_records.csv --mapper data/tillage_mapper_input.csv --out out --debug
```

This will sequentially run:

1. `wq_longify.py` (if not skipped),
2. `stir_pipeline.py`,
3. `merge_wq_stir_by_season.py`,
   and write outputs to your `/out` directory.

---

If you want to skip the preprocessing steps and only re-run the merge:

```bash
python code/main.py --skip-wq --skip-stir --crops "data/crop records.csv" --out out --debug
```

**From a Python prompt (one function):**

```python
from main import run_all
run_all(debug=True)
```