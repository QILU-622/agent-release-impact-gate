import pytest

from agent_mesh_risk_lab.benchmark import generate_benchmark
from agent_mesh_risk_lab.portfolio_experiments import (
    all_control_subsets,
    optimize_empirical_portfolio,
    portfolio_id,
    run_control_portfolio_grid,
    shapley_control_value,
)


def test_all_control_subsets_are_complete_and_unique():
    subsets = all_control_subsets()
    assert len(subsets) == 64
    assert len({portfolio_id(subset) for subset in subsets}) == 64


@pytest.fixture(scope="module")
def small_grid():
    tasks = generate_benchmark()
    representative_tasks = [tasks[index] for index in [0, 50, 100, 150]]
    return run_control_portfolio_grid(representative_tasks)


def test_empirical_optimizer_respects_observed_guardrails(small_grid):
    grid, _ = small_grid
    result = optimize_empirical_portfolio(grid, budget=40)
    assert result["cost"] <= 40
    assert result["task_success_rate"] >= 0.85
    assert result["human_review_load"] <= 0.30
    assert result["risk_reduction"] > 0


def test_shapley_values_reconcile_to_full_coalition_reduction(small_grid):
    grid, _ = small_grid
    shapley = shapley_control_value(grid)
    baseline = float(grid.loc[grid["portfolio"] == "none", "incident_rate"].iloc[0])
    full_name = portfolio_id(all_control_subsets()[-1])
    full_risk = float(grid.loc[grid["portfolio"] == full_name, "incident_rate"].iloc[0])
    assert abs(shapley["shapley_risk_reduction"].sum() - (baseline - full_risk)) < 1e-10
    assert shapley["efficiency_gap"].abs().max() < 1e-10


def test_small_portfolio_grid_has_expected_shape(small_grid):
    grid, workflow = small_grid
    assert len(grid) == 64
    assert len(workflow) == 64 * 4
