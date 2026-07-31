from __future__ import annotations

import json
import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
STAN_PATH = REPO / "code" / "bayes" / "m_stir_mogp_v3p2_physical_event.stan"
BATCH_PATH = (
    REPO / "code" / "bayes" / "stir-bayes-load_v3p2_physical_event.R"
)
RMD_PATH = (
    REPO / "code" / "bayes" / "stir-bayes-load_v3p2_physical_event.Rmd"
)
CONFIG_PATH = REPO / "config" / "physical_event_v3p2.json"
WORKFLOW_PATH = REPO / "docs" / "workflows" / "bayes_v3p2_physical_event.md"
PRIOR_AUDIT_PATH = REPO / "docs" / "methods" / "bayesian_prior_audit_v3p2.md"
SAVED_FIT_DIAGNOSTIC_PATH = (
    REPO / "code" / "bayes" / "diagnose_saved_fit_v3p2.R"
)


def test_v3p2_release_files_and_output_isolation() -> None:
    for path in (
        STAN_PATH,
        BATCH_PATH,
        RMD_PATH,
        CONFIG_PATH,
        WORKFLOW_PATH,
        PRIOR_AUDIT_PATH,
        SAVED_FIT_DIAGNOSTIC_PATH,
    ):
        assert path.is_file(), path

    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    assert config["workflow_version"] == "v3p2_physical_event"
    assert config["versions"] == {
        "bayesian": "v3p2_physical_event",
        "ml": "v3p1_physical_event",
        "comparison": "v3p2_physical_event",
    }
    assert config["output_roots"]["bayesian_results"] == (
        "results/bayes/v3p2_physical_event"
    )
    assert config["output_roots"]["bayesian_figures"] == (
        "figures/bayes/v3p2_physical_event"
    )
    assert config["output_roots"]["comparison_results"] == (
        "results/comparison/v3p2_physical_event"
    )
    assert config["output_roots"]["comparison_figures"] == (
        "figures/comparison/v3p2_physical_event"
    )
    assert config["data_contract"] == {
        "workflow_version": "v3p1_physical_event",
        "preflight_directory": "validation/preflight/physical_event_v3p1",
        "reason": (
            "v3p2 changes only the Bayesian parameterization and priors; "
            "the corrected 528-event input and preflight contract are unchanged"
        ),
    }


def test_v3p2_uses_one_global_volume_effect_on_concentration() -> None:
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


def test_v3p2_requested_unit_prior_conventions_are_complete() -> None:
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
        "beta_res_C",
        "mu_A",
        "b_res_stir",
    }
    for parameter in required_unit_normals:
        assert f"{parameter} ~ normal(0, 1);" in stan

    assert "mu_log_sigma_C_obs ~ normal(log(0.2), 1);" in stan
    assert "res_base ~ beta(2, 10);" in stan
    assert "L_corr_C_event ~ lkj_corr_cholesky(2);" in stan


def test_v3p2_batch_extracts_and_reports_scalar_beta_vol() -> None:
    batch = BATCH_PATH.read_text(encoding="utf-8")

    assert 'file.path(repo_root, "config", "physical_event_v3p2.json")' in batch
    assert '"m_stir_mogp_v3p2_physical_event.stan"' in batch
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


def test_v3p2_rstudio_and_workflow_entrypoints_are_current() -> None:
    rmd = RMD_PATH.read_text(encoding="utf-8")
    workflow = WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "stir-bayes-load_v3p2_physical_event.R" in rmd
    assert "source(batch_script, chdir = FALSE)" in rmd
    assert "one global standardized direct effect" in workflow
    assert "No pipeline or Python data" in workflow
    assert "bayesian_prior_audit_v3p2.md" in workflow


def test_v3p2_saved_fit_residue_audit_includes_global_volume_path() -> None:
    diagnostic = SAVED_FIT_DIAGNOSTIC_PATH.read_text(encoding="utf-8")

    assert "fit_mogp_v3p2_physical_event.rds" in diagnostic
    assert "mod$sample(" not in diagnostic
    assert "cmdstan_model(" not in diagnostic
    assert '"beta_res_V", "a_V", "b_V", "beta_vol", "beta_vin"' in diagnostic


def test_v3p2_annual_single_plot_markers_and_volume_completeness() -> None:
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
