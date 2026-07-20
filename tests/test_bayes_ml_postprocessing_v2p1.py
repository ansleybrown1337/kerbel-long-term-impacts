from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

import bayes_ml_postprocessing_v2p1 as pp  # noqa: E402


OUT = REPO / "out" / "bayes_vs_ml_postprocessing_v2p1"


class UnitTests(unittest.TestCase):
    def test_analyte_alias_mapping(self) -> None:
        expected = {
            "NH4-N": "NH4", "Ammonium(NH4)": "NH4", "Nitrate": "NO3",
            "Nitrite": "NO2", "OrthoP": "OP", "Selenium": "Se",
            "TDS": "TDS", "TKN-N": "TKN", "TotalN": "TN",
            "TotalP": "TP", "TSS": "TSS",
        }
        self.assertEqual({key: pp.canonical_analyte(key) for key in expected}, expected)

    def test_required_columns_and_unique_keys(self) -> None:
        frame = pd.DataFrame({"draw": [1, 2], "Year": [2011, 2011]})
        pp.require_columns(frame, ["draw", "Year"], "fixture")
        pp.assert_unique(frame, ["draw", "Year"], "fixture")
        with self.assertRaises(ValueError):
            pp.require_columns(frame, ["load_g"], "fixture")
        with self.assertRaises(ValueError):
            pp.assert_unique(pd.concat([frame, frame.iloc[[0]]]), ["draw", "Year"], "fixture")

    def test_unit_conversions(self) -> None:
        np.testing.assert_allclose(pp.grams_to_kg(np.array([1_000.0, -500.0])), [1.0, -0.5])
        np.testing.assert_allclose(pp.milligrams_to_kg(np.array([1_000_000.0, 500.0])), [1.0, 0.0005])

    def test_expected_year_reporting(self) -> None:
        self.assertEqual(pp.missing_years([2011, 2012, 2025]), list(range(2013, 2025)))
        self.assertEqual(pp.missing_year_text(pp.STUDY_YEARS), "None")

    def test_hand_calculated_cumulative_and_bayes_variants(self) -> None:
        rows = []
        values = {
            "CT": {1: [1000.0, -500.0], 2: [2000.0, 500.0]},
            "MT": {1: [500.0, -100.0], 2: [1000.0, 100.0]},
            "ST": {1: [750.0, -200.0], 2: [1500.0, 200.0]},
        }
        for treatment, draw_values in values.items():
            for draw, annual in draw_values.items():
                for year, load in zip((2011, 2012), annual):
                    rows.append({"draw": draw, "draw_id": draw + 100, "Year": year, "analyte": "NH4", "treatment": treatment, "load_g": load})
        summary, draws = pp.bayes_cumulative_products(pd.DataFrame(rows), Path("fixture.csv"))
        np.testing.assert_allclose(draws[("raw_draws_bound_floor", "NH4", "CT")], [0.5, 2.5])
        np.testing.assert_allclose(draws[("annual_draw_truncation", "NH4", "CT")], [1.0, 2.5])
        raw = summary.loc[(summary.sensitivity_variant == "raw_draws_bound_floor") & (summary.treatment == "CT")].iloc[0]
        self.assertAlmostEqual(raw["mean_cumulative_load_kg"], 1.5)

    def test_variant_a_display_floor_does_not_change_raw_bound(self) -> None:
        values = np.array([-2.0, -1.0, 1.0, 2.0])
        summary = pp.summarize_draws(values)
        displayed = max(summary["low"], 0.0)
        self.assertLess(summary["low"], 0.0)
        self.assertEqual(displayed, 0.0)
        np.testing.assert_array_equal(values, np.array([-2.0, -1.0, 1.0, 2.0]))

    def test_ct_alignment_and_invalid_denominators(self) -> None:
        values = {}
        for analyte in pp.ANALYTES:
            values[("fixture", analyte, "CT")] = np.array([1.0, 0.0, -1.0, 1e-13])
            values[("fixture", analyte, "MT")] = np.array([0.5, 0.1, 0.5, 0.0])
            values[("fixture", analyte, "ST")] = np.array([0.75, 0.2, 0.5, 0.0])
        result = pp.treatment_differences("Bayes", ["fixture"], values, "fixture")
        row = result.loc[(result.analyte == "NH4") & (result.comparison_treatment == "MT")].iloc[0]
        self.assertEqual(row["n_valid_percent_draws"], 1)
        self.assertEqual(row["n_invalid_zero_ct_draws"], 1)
        self.assertEqual(row["n_invalid_negative_ct_draws"], 1)
        self.assertEqual(row["n_invalid_tiny_positive_ct_draws"], 1)
        self.assertAlmostEqual(row["mean_percent_difference_relative_to_ct"], 50.0)


@unittest.skipUnless(OUT.exists(), "Run post-processing integration workflow first")
class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cumulative = pd.read_csv(OUT / "study_period_cumulative_loads_raw.csv", keep_default_na=False)
        cls.differences = pd.read_csv(OUT / "treatment_differences_vs_ct_raw.csv")

    def test_expected_groups_treatments_and_years(self) -> None:
        self.assertEqual(len(self.cumulative), 90)
        self.assertEqual(set(self.cumulative.treatment), set(pp.TREATMENTS))
        self.assertTrue((self.cumulative.n_included_years == 15).all())
        self.assertTrue((self.cumulative.missing_years == "None").all())

    def test_saved_summary_reconciliation_status(self) -> None:
        recon = pd.read_csv(OUT / "annual_summary_reconciliation.csv")
        self.assertTrue(recon.loc[recon.method == "ML", "exact_reconciliation"].all())
        self.assertFalse(recon.loc[recon.method == "Bayes", "exact_reconciliation"].all())

    def test_variant_a_and_b_against_saved_bayes_draws(self) -> None:
        source = pd.read_csv(REPO / "out" / "annual_load_draws_bayes_v2p1.csv")
        group = source.loc[(source.analyte == "NO2") & (source.treatment == "CT")]
        raw = group.groupby("draw").load_g.sum().to_numpy() / 1000
        truncated = group.assign(x=group.load_g.clip(lower=0)).groupby("draw").x.sum().to_numpy() / 1000
        a = self.cumulative.loc[(self.cumulative.analyte == "NO2") & (self.cumulative.treatment == "CT") & (self.cumulative.sensitivity_variant == "raw_draws_bound_floor")].iloc[0]
        b = self.cumulative.loc[(self.cumulative.analyte == "NO2") & (self.cumulative.treatment == "CT") & (self.cumulative.sensitivity_variant == "annual_draw_truncation")].iloc[0]
        self.assertAlmostEqual(a.mean_cumulative_load_kg, raw.mean())
        self.assertAlmostEqual(a.lower_95_raw_kg, np.quantile(raw, 0.025))
        self.assertEqual(a.lower_95_display_kg, 0.0)
        self.assertAlmostEqual(b.mean_cumulative_load_kg, truncated.mean())

    def test_spearman_sample_sizes_and_markers(self) -> None:
        raw = pd.read_csv(OUT / "spearman_by_analyte_treatment_raw.csv", keep_default_na=False)
        self.assertEqual(len(raw), 30)
        self.assertTrue((raw.n_years == 15).all())
        expected = np.where(raw.p_value_unadjusted < 0.05, "*", "")
        np.testing.assert_array_equal(raw.significance_marker.to_numpy(), expected)

    def test_publication_master_matches_raw_source(self) -> None:
        master = pd.read_csv(OUT / "master_cumulative_loads_raw_bound_floor_pub.csv")
        raw = self.cumulative.loc[(self.cumulative.method == "Bayes") & (self.cumulative.sensitivity_variant == "raw_draws_bound_floor") & (self.cumulative.analyte == "NH4") & (self.cumulative.treatment == "CT")].iloc[0]
        cell = master.loc[master.Analyte == "NH4", "Bayes CT, kg [95% interval]"].iloc[0]
        self.assertEqual(cell, pp.format_load(raw.mean_cumulative_load_kg, raw.lower_95_display_kg, raw.upper_95_kg))

    def test_new_paths_do_not_target_existing_v2p1_artifacts(self) -> None:
        self.assertNotEqual(OUT.name, "bayes_vs_ml_metrics_v2p1")
        self.assertTrue((REPO / "out" / "bayes_vs_ml_metrics_v2p1").exists())


if __name__ == "__main__":
    unittest.main()
