from __future__ import annotations

import json
import csv
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
STAN_PATH = REPO / "code" / "bayes" / "m_stir_mogp_v3p3_physical_event.stan"
BATCH_PATH = (
    REPO / "code" / "bayes" / "stir-bayes-load_v3p3_physical_event.R"
)
RMD_PATH = (
    REPO / "code" / "bayes" / "stir-bayes-load_v3p3_physical_event.Rmd"
)
CONFIG_PATH = REPO / "config" / "physical_event_v3p3.json"
WORKFLOW_PATH = REPO / "docs" / "workflows" / "bayes_v3p3_physical_event.md"
PRIOR_AUDIT_PATH = REPO / "docs" / "methods" / "bayesian_prior_audit_v3p3.md"
SAVED_FIT_DIAGNOSTIC_PATH = (
    REPO / "code" / "bayes" / "diagnose_saved_fit_v3p3.R"
)
PERIOD_TABLE_HELPER_PATH = (
    REPO / "code" / "bayes" / "period_total_load_tables.R"
)
PERIOD_TABLE_RUNNER_PATH = (
    REPO
    / "code"
    / "bayes"
    / "export_period_total_load_tables_v3p3_physical_event.R"
)
COMPACTION_PATH = REPO / "data" / "furrow_tire_compaction_records.csv"
PREFLIGHT_EVENTS_PATH = (
    REPO
    / "validation"
    / "preflight"
    / "physical_event_v3p1"
    / "physical_events.csv"
)


def test_v3p3_release_files_and_output_isolation() -> None:
    for path in (
        STAN_PATH,
        BATCH_PATH,
        RMD_PATH,
        CONFIG_PATH,
        WORKFLOW_PATH,
        PRIOR_AUDIT_PATH,
        SAVED_FIT_DIAGNOSTIC_PATH,
        PERIOD_TABLE_HELPER_PATH,
        PERIOD_TABLE_RUNNER_PATH,
        COMPACTION_PATH,
    ):
        assert path.is_file(), path

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["workflow_version"] == "v3p3_physical_event"
    assert config["versions"] == {
        "bayesian": "v3p3_physical_event",
        "ml": "v3p3_physical_event",
        "comparison": "v3p3_physical_event",
    }
    assert config["output_roots"]["bayesian_results"] == (
        "results/bayes/v3p3_physical_event"
    )
    assert config["output_roots"]["bayesian_figures"] == (
        "figures/bayes/v3p3_physical_event"
    )
    assert config["output_roots"]["ml_results"] == (
        "results/ml/v3p3_physical_event"
    )
    assert config["output_roots"]["ml_figures"] == (
        "figures/ml/v3p3_physical_event"
    )
    assert config["output_roots"]["comparison_results"] == (
        "results/comparison/v3p3_physical_event"
    )
    assert config["output_roots"]["comparison_figures"] == (
        "figures/comparison/v3p3_physical_event"
    )
    assert config["data_contract"] == {
        "workflow_version": "v3p1_physical_event",
        "preflight_directory": "validation/preflight/physical_event_v3p1",
        "reason": (
            "v3p3 adds the documented event-level furrow tire-compaction "
            "predictor to the Bayesian and ML runoff-volume models only; the "
            "corrected 528-event input and v3p1 preflight contract are "
            "unchanged"
        ),
    }
    assert config["furrow_tire_compaction"] == {
        "source_file": "data/furrow_tire_compaction_records.csv",
        "assignment_key": ["Year", "Treatment"],
        "event_field": "FurrowTireCompaction",
        "default_unexposed_value": 0,
        "affected_replicates": "both",
        "timing": "after residue measurement and before the first runoff event",
        "model_scope": "runoff volume only",
        "carryover": (
            "none; the indicator applies to runoff events in the documented "
            "treatment-year"
        ),
        "expected_compacted_events": 120,
        "expected_compacted_events_with_genuine_volume": 70,
    }


def test_v3p3_uses_one_global_volume_effect_on_concentration() -> None:
    stan = STAN_PATH.read_text(encoding="utf-8")
    parameters = stan.split("parameters {", 1)[1].split(
        "transformed parameters {", 1
    )[0]
    concentration_process = stan.split(
        "mu_C_event[e, a] = alpha[a]", 1
    )[1].split("C_true_event =", 1)[0]

    assert "real beta_vol;" in parameters
    assert "vector[A_n] beta_vol;" not in stan
    assert "mu_beta_vol" not in stan
    assert "sigma_beta_vol" not in stan
    assert "z_beta_vol" not in stan
    assert "+ beta_vol * V_true_event[e]" in concentration_process
    assert "beta_vol[a]" not in stan


def test_v3p3_requested_unit_prior_conventions_are_complete() -> None:
    stan = STAN_PATH.read_text(encoding="utf-8")

    exponential_rates = re.findall(r"~\s*exponential\(([^)]+)\)", stan)
    assert exponential_rates
    assert set(exponential_rates) == {"1"}

    numeric_zero_centered_normal_sds = [
        float(value)
        for value in re.findall(r"~\s*normal\(0,\s*([0-9.]+)\)", stan)
    ]
    assert numeric_zero_centered_normal_sds
    assert set(numeric_zero_centered_normal_sds) == {1.0}

    required_unit_normals = {
        "a_V",
        "b_V",
        "beta_vol",
        "beta_vin",
        "mu_beta_irr",
        "mu_beta_dup",
        "beta_res_V",
        "beta_tire_comp_V",
        "beta_res_C",
        "mu_A",
        "b_res_stir",
    }
    for parameter in required_unit_normals:
        assert f"{parameter} ~ normal(0, 1);" in stan

    assert "mu_log_sigma_C_obs ~ normal(log(0.2), 1);" in stan
    assert "res_base ~ beta(2, 10);" in stan
    assert "L_corr_C_event ~ lkj_corr_cholesky(2);" in stan


def test_v3p3_batch_extracts_and_reports_scalar_beta_vol() -> None:
    batch = BATCH_PATH.read_text(encoding="utf-8")

    assert 'file.path(repo_root, "config", "physical_event_v3p3.json")' in batch
    assert '"m_stir_mogp_v3p3_physical_event.stan"' in batch
    assert "beta_vol_is_global_scalar" in batch
    assert "beta_vol_hierarchy_removed" in batch
    assert 'get_param_vec(draws_mat, "beta_vol")' in batch
    assert 'get_param_mat(draws_mat, "beta_vol")' not in batch
    assert 'vnames_vec("beta_vol"' not in batch
    assert "post$beta_vol[draw_idx]" in batch
    assert "post$beta_vol[draw_idx, j_idx]" not in batch
    assert '"beta_vol", "Global runoff-volume effect on concentration"' in batch
    assert "Global runoff volume -> concentration" in batch
    assert "data_contract_version" in batch
    assert (
        "preflight_metadata$workflow_version, data_contract_version" in batch
    )
    assert 'runoff_volume_effect_on_concentration = "one global standardized beta_vol"' in batch
    assert 'standardized_zero_centered_coefficients = "normal(0, 1)"' in batch
    assert 'positive_scale_parameters = "exponential(1)"' in batch
    assert "results/bayes/v3p1_physical_event" not in batch


def test_v3p3_rstudio_and_workflow_entrypoints_are_current() -> None:
    rmd = RMD_PATH.read_text(encoding="utf-8")
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "stir-bayes-load_v3p3_physical_event.R" in rmd
    assert "source(batch_script, chdir = FALSE)" in rmd
    assert "FurrowTireCompaction" in workflow
    assert "runoff-volume process only" in workflow
    assert "No pipeline or Python data" in workflow
    assert "bayesian_prior_audit_v3p3.md" in workflow


def test_v3p3_saved_fit_residue_audit_includes_global_volume_path() -> None:
    diagnostic = SAVED_FIT_DIAGNOSTIC_PATH.read_text(encoding="utf-8")

    assert "fit_mogp_v3p3_physical_event.rds" in diagnostic
    assert "mod$sample(" not in diagnostic
    assert "cmdstan_model(" not in diagnostic
    assert '"beta_res_V", "beta_tire_comp_V", "a_V", "b_V", "beta_vol"' in diagnostic
    assert "tire_compaction_parameter_correlations_v3p3_physical_event.csv" in diagnostic


def test_v3p3_annual_single_plot_markers_and_volume_completeness() -> None:
    batch = BATCH_PATH.read_text(encoding="utf-8")
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert '"Observed (1 complete plot)" = 22' in batch
    assert '"Observed (replicate range)" = 21' in batch
    assert "One complete plot: white square, no interval" in batch
    assert "shape = 4" not in batch
    assert "n_complete_volume_plots = as.integer(n_complete_volume_plots)" in batch
    annual_plot_saver = batch.split(
        "save_all_analyte_obs_vs_mod_plots <- function(", 1
    )[1].split("# Example:", 1)[0]
    assert "width = 12," in annual_plot_saver
    assert 'identical(analyte_pick, "TSS")' in batch
    assert "1 / 1e6" in batch
    assert '"Annual load (Mg)"' in batch
    assert "saved annual" in batch
    assert (
        "n_complete_volume_plots = if "
        "(all(is.na(n_complete_volume_plots)))"
    ) in batch
    assert ".data$n_complete_volume_plots >= 2" in batch
    assert ".data$n_complete_volume_plots == 1" in batch
    assert (
        'observed_single_complete_plot_marker = '
        '"white-filled square with treatment-colored outline"'
    ) in batch
    assert "white-filled square" in workflow
    assert "volume-specific complete-plot count" in workflow


def test_v3p3_compaction_source_has_exact_approved_scope() -> None:
    with COMPACTION_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    observed = {(int(row["Year"]), row["Treatment"]) for row in rows}
    expected = {
        (2021, "ST"),
        (2022, "MT"),
        (2022, "ST"),
        (2023, "MT"),
        (2023, "ST"),
        (2024, "MT"),
        (2024, "ST"),
        (2025, "MT"),
        (2025, "ST"),
    }
    assert observed == expected
    assert len(rows) == len(expected)
    assert all(row["RepScope"] == "both" for row in rows)
    assert all(row["FurrowTireCompaction"] == "1" for row in rows)
    assert all(
        row["Timing"] == "after_residue_before_first_runoff"
        for row in rows
    )
    rows_2024 = [row for row in rows if row["Year"] == "2024"]
    assert len(rows_2024) == 2
    assert all(row["AnchorDate"] == "" for row in rows_2024)
    assert all(
        row["DateStatus"] == "user_confirmed_before_first_runoff"
        for row in rows_2024
    )


def test_v3p3_compaction_event_support_matches_reviewed_roster() -> None:
    with COMPACTION_PATH.open(newline="", encoding="utf-8") as handle:
        assignments = {
            (int(row["Year"]), row["Treatment"])
            for row in csv.DictReader(handle)
        }
    with PREFLIGHT_EVENTS_PATH.open(newline="", encoding="utf-8") as handle:
        events = list(csv.DictReader(handle))

    compacted = [
        row
        for row in events
        if (int(row["Year"]), row["Treatment"]) in assignments
    ]
    compacted_with_volume = [
        row
        for row in compacted
        if int(row["genuine_volume_observations"]) > 0
    ]
    assert len(events) == 528
    assert len(compacted) == 120
    assert len(compacted_with_volume) == 70
    assert not [
        row
        for row in compacted
        if row["Treatment"] == "CT" or int(row["Year"]) < 2021
    ]


def test_v3p3_compaction_enters_only_event_volume_process() -> None:
    stan = STAN_PATH.read_text(encoding="utf-8")
    batch = BATCH_PATH.read_text(encoding="utf-8")
    residue_process = stan.split("for (u in 1:R_n)", 1)[1].split(
        "for (e in 1:E_n)", 1
    )[0]
    volume_process = stan.split("for (e in 1:E_n)", 1)[1].split(
        "V_true_event =", 1
    )[0]
    concentration_process = stan.split(
        "mu_C_event[e, a] = alpha[a]", 1
    )[1].split("C_true_event =", 1)[0]

    assert (
        "array[E_n] int<lower=0, upper=1> TIRE_COMPACTION_E;"
        in stan
    )
    assert "real beta_tire_comp_V;" in stan
    assert "beta_tire_comp_V ~ normal(0, 1);" in stan
    assert (
        "+ beta_tire_comp_V * TIRE_COMPACTION_E[e]"
        in volume_process
    )
    assert "beta_tire_comp_V" not in residue_process
    assert "beta_tire_comp_V" not in concentration_process
    assert stan.count(
        "beta_tire_comp_V * TIRE_COMPACTION_E[e]"
    ) == 1

    assert (
        "compaction_path <- file.path("
        "repo_root, compaction_config$source_file)"
        in batch
    )
    assert "TIRE_COMPACTION_E" in batch
    assert "expected_compacted_events" in batch
    assert "expected_compacted_events_with_genuine_volume" in batch
    assert (
        "tire_compaction_event_audit_v3p3_physical_event.csv"
        in batch
    )
    assert (
        "tire_compaction_audit_summary_v3p3_physical_event.csv"
        in batch
    )
    assert 'get_param_vec(draws_mat, "beta_tire_comp_V")' in batch
    assert (
        '"beta_tire_comp_V", '
        '"Furrow tire-compaction effect on runoff volume"'
        in batch
    )
    assert "tire_compaction_annual_volume_contrast" in batch


def test_v3p3_period_total_exports_use_saved_replicate_aware_draws() -> None:
    batch = BATCH_PATH.read_text(encoding="utf-8")
    helper = PERIOD_TABLE_HELPER_PATH.read_text(encoding="utf-8")
    runner = PERIOD_TABLE_RUNNER_PATH.read_text(encoding="utf-8")

    assert 'source(file.path(repo_root, "code", "bayes", "period_total_load_tables.R"))' in batch
    assert '"pre_tire_compaction_era"' in batch
    assert '"tire_compaction_era"' in batch
    assert "2020L" in batch
    assert "2021L" in batch
    assert "2025L" in batch

    assert '"\\n("' in helper
    assert '"%\\n("' in helper
    assert "group_by(\n      .data$draw, .data$draw_id, .data$analyte, .data$treatment" in helper
    assert "load_sum_kg = sum(.data$load_g) / 1000" in helper
    assert "MT_pct_reduction_from_CT = 100 * (1 - .data$MT / .data$CT)" in helper
    assert "ST_pct_reduction_from_CT = 100 * (1 - .data$ST / .data$CT)" in helper

    assert 'paste0("annual_load_draws_bayes_", model_version, ".csv")' in runner
    assert "readr::read_csv(" in runner
    assert "fit_mogp" not in runner
    assert "cmdstan_model(" not in runner
    assert "mod$sample(" not in runner
