from pathlib import Path

import pandas as pd

from app.cleaning.detector import detect_issues
from app.cleaning.planner import build_cleaning_plan


def test_cleaning_plan():
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sample"
        / "orders.csv"
    )

    df = pd.read_csv(dataset_path)

    issues = detect_issues(df)
    plan = build_cleaning_plan(issues)

    assert len(plan) >= 3

    step_ids = {step["step_id"] for step in plan}

    assert "CLEAN-001" in step_ids
    assert "CLEAN-002" in step_ids
    assert "CLEAN-003" in step_ids

    for step in plan:
        assert step["issue_id"]
        assert step["reason"]
        assert step["fields"]
        assert step["risk"]
        assert step["source"]
        assert step["status"] == "planned"
        