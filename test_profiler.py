from pathlib import Path

from app.profiling.profiler import profile_csv


def test_orders_profile():
    dataset_path = (
        Path(__file__).resolve().parents[1]
        / "data"
        / "sample"
        / "orders.csv"
    )

    profile = profile_csv(dataset_path)

    assert profile["dataset"] == "orders.csv"
    assert profile["rows"] == 6
    assert profile["columns"] == 9

    assert profile["duplicates"]["duplicate_rows"] == 2
    assert profile["cardinality"]["order_id"] == 5