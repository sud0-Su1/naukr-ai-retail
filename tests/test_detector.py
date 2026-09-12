from pathlib import Path

import pandas as pd

from app.cleaning.detector import detect_issues


def test_orders_issue_detection():
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sample"
        / "orders.csv"
    )

    df = pd.read_csv(dataset_path)

    issues = detect_issues(df)

    issue_types = {issue["type"] for issue in issues}

    assert "duplicate_rows" in issue_types
    assert "numeric_formatting" in issue_types
    assert "date_formatting" in issue_types
