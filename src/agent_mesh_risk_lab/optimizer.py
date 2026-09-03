"""Budget-constrained governance portfolio optimizer."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from .catalog import CONTROLS


def optimize_portfolio(
    governance_roi: pd.DataFrame,
    budget: float = 40,
    min_completion: float = 0.85,
    max_review_load: float = 0.30,
) -> dict:
    """Enumerate all 64 portfolios; effects use measured single-control counterfactuals."""
    if governance_roi.empty:
        return {"controls": [], "cost": 0.0, "risk_reduction": 0.0}
    records = governance_roi.set_index("control").to_dict("index")
    base_risk = float(governance_roi["risk_before"].iloc[0])
    base_completion = float(governance_roi["completion_before"].iloc[0])
    base_review = float(governance_roi["baseline_review_load"].iloc[0])
    best: dict | None = None
    controls = list(CONTROLS)
    for size in range(len(controls) + 1):
        for combo in combinations(controls, size):
            cost = sum(float(CONTROLS[name]["cost"]) for name in combo)
            if cost > budget:
                continue
            residual_fraction = 1.0
            for name in combo:
                relative = records[name]["risk_reduction"] / base_risk if base_risk else 0.0
                residual_fraction *= 1 - max(0.0, min(0.95, relative))
            reduction = base_risk * (1 - residual_fraction)
            completion = base_completion - sum(
                float(CONTROLS[name]["completion_penalty"]) for name in combo
            )
            review = base_review + sum(float(CONTROLS[name]["review_add"]) for name in combo)
            if completion < min_completion or review > max_review_load:
                continue
            candidate = {
                "controls": list(combo),
                "cost": round(cost, 2),
                "risk_reduction": round(reduction, 6),
                "residual_risk": round(max(0.0, base_risk - reduction), 6),
                "estimated_completion": round(completion, 6),
                "estimated_review_load": round(review, 6),
                "method": "exhaustive search over measured single-control effects",
            }
            if best is None or (candidate["risk_reduction"], -candidate["cost"]) > (
                best["risk_reduction"],
                -best["cost"],
            ):
                best = candidate
    return best or {
        "controls": [],
        "cost": 0.0,
        "risk_reduction": 0.0,
        "residual_risk": base_risk,
        "estimated_completion": base_completion,
        "estimated_review_load": base_review,
        "method": "no feasible portfolio",
    }


def budget_curve(
    governance_roi: pd.DataFrame, max_budget: int = 100, step: int = 5
) -> pd.DataFrame:
    rows = []
    for budget in range(0, max_budget + 1, step):
        result = optimize_portfolio(governance_roi, budget=budget)
        rows.append({"budget": budget, **result, "controls": ", ".join(result["controls"])})
    return pd.DataFrame(rows)
