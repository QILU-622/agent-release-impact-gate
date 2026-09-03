import pandas as pd

from agent_mesh_risk_lab.benchmark import generate_benchmark
from agent_mesh_risk_lab.features import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_MODEL_COLUMNS,
    NUMERIC_FEATURES,
    TEXT_FEATURES,
    assign_task_splits,
    build_feature_frame,
)
from agent_mesh_risk_lab.simulator import run_experiment


def test_task_group_split_has_no_overlap_and_preserves_all_tasks():
    tasks = generate_benchmark()
    split_map = assign_task_splits(tasks)
    groups = {
        split: {task_id for task_id, assigned in split_map.items() if assigned == split}
        for split in ["train", "validation", "test"]
    }
    assert len(split_map) == 200
    assert not (groups["train"] & groups["validation"])
    assert not (groups["train"] & groups["test"])
    assert not (groups["validation"] & groups["test"])
    assert {name: len(values) for name, values in groups.items()} == {
        "train": 136,
        "validation": 32,
        "test": 32,
    }


def test_feature_frame_excludes_post_action_outcomes():
    tasks = generate_benchmark()[:4]
    rows = []
    for task in tasks:
        run = run_experiment(task, "policy_drop", ["context_envelope"])
        payload = run.model_dump()
        payload.pop("trace")
        payload["controls"] = ",".join(payload["controls"])
        rows.append(payload)
    frame = build_feature_frame(tasks, pd.DataFrame(rows))
    model_columns = set(NUMERIC_FEATURES + CATEGORICAL_FEATURES + TEXT_FEATURES)
    assert model_columns <= set(frame.columns)
    assert not model_columns & FORBIDDEN_MODEL_COLUMNS
    assert frame["harmful_label"].isin([0, 1]).all()
