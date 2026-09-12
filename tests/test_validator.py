from pathlib import Path

import pandas as pd

from app.cleaning.cleaner import clean_orders
from app.cleaning.detector import detect_issues
from app.cleaning.planner import build_cleaning_plan
from app.cleaning.validator import (
    check_idempotency,
    validate_cleaning,
)


def get_cleaning_result():
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sample"
        / "orders.csv"
    )

    df = pd.read_csv(dataset_path)

    issues = detect_issues(df)
    plan = build_cleaning_plan(issues)

    cleaned, updated_plan = clean_orders(
        df,
        plan,
    )

    return df, cleaned, updated_plan


def test_cleaning_validation():
    before, after, _ = get_cleaning_result()

    validation = validate_cleaning(
        before,
        after,
    )

    assert validation["valid"] is True
    assert validation["rows"]["before"] == 6
    assert validation["rows"]["after"] == 5
    assert validation["rows"]["removed"] == 1
    assert validation["schema"]["unchanged"] is True


def test_cleaning_idempotency():
    before, cleaned, plan = get_cleaning_result()

    result = check_idempotency(
        clean_orders,
        cleaned,
        plan,
    )

    assert result["same_shape"] is True
    assert result["same_columns"] is True
    assert result["same_data"] is True
    assert result["idempotent"] is True