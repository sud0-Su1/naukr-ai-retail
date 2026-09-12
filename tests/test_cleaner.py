from pathlib import Path

import pandas as pd

from app.cleaning.cleaner import clean_orders
from app.cleaning.detector import detect_issues
from app.cleaning.planner import build_cleaning_plan


def test_orders_cleaning():
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sample"
        / "orders.csv"
    )

    df = pd.read_csv(dataset_path)

    issues = detect_issues(df)
    plan = build_cleaning_plan(issues)

    cleaned, updated_plan = clean_orders(df, plan)

    # Duplicate row removed.
    assert len(cleaned) == 5

    # unit_price converted to numeric.
    assert pd.api.types.is_numeric_dtype(
        cleaned["unit_price"]
    )

    # Dates normalized.
    assert cleaned["order_date"].notna().all()

    assert set(
        cleaned["order_date"].astype(str)
    ) == {
        "2025-01-10",
        "2025-02-10",
        "2025-02-15",
        "2025-01-03",
        "2025-03-12",
    }

    # All cleaning steps completed.
    assert all(
        step["status"] == "applied"
        for step in updated_plan
    )