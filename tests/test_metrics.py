from pathlib import Path

import pandas as pd
import pytest

from app.cleaning.cleaner import clean_orders
from app.cleaning.detector import detect_issues
from app.cleaning.planner import build_cleaning_plan
from app.retail.metrics import (
    calculate_aov,
    calculate_inventory_stockout_frequency,
    calculate_return_rate,
    calculate_revenue,
    sales_by_category,
    sales_by_store,
)


BASE_DIR = Path(__file__).resolve().parents[1]


def get_clean_orders() -> pd.DataFrame:
    path = BASE_DIR / "data" / "sample" / "orders.csv"

    df = pd.read_csv(path)

    issues = detect_issues(df)
    plan = build_cleaning_plan(issues)

    cleaned, _ = clean_orders(df, plan)

    return cleaned


def get_products() -> pd.DataFrame:
    return pd.read_csv(
        BASE_DIR / "data" / "sample" / "products.csv"
    )


def get_stores() -> pd.DataFrame:
    return pd.read_csv(
        BASE_DIR / "data" / "sample" / "stores.csv"
    )


def get_inventory() -> pd.DataFrame:
    return pd.read_csv(
        BASE_DIR / "data" / "sample" / "inventory.csv"
    )


def test_revenue():
    orders = get_clean_orders()

    # Manual expected value:
    #
    # ORD001 = 2 * 1299 * 0.90 = 2338.2
    # ORD002 = 1 * 999 * 1.00 = 999
    # ORD003 = 3 * 2499 * 0.95 = 7122.15
    # ORD004 = 1 * 1999 * 0.85 = 1699.15
    # ORD005 = 2 * 1799 * 0.90 = 3238.2
    #
    # Total = 15396.70

    assert calculate_revenue(orders) == pytest.approx(
        15396.70
    )


def test_aov():
    orders = get_clean_orders()

    expected_aov = 15396.70 / 5

    assert calculate_aov(orders) == pytest.approx(
        expected_aov
    )


def test_return_rate():
    orders = get_clean_orders()

    # ORD002 and ORD005 are returned.
    assert calculate_return_rate(orders) == pytest.approx(
        2 / 5
    )


def test_sales_by_category():
    orders = get_clean_orders()
    products = get_products()

    result = sales_by_category(
        orders,
        products,
    )
    categories = set(
        result["category"].dropna()
    )

    assert "T-Shirt" in categories
    assert "tee" not in categories

    assert "category" in result.columns
    assert "revenue" in result.columns

    assert result["revenue"].sum() == pytest.approx(
        15396.70
    )


def test_sales_by_store():
    orders = get_clean_orders()
    stores = get_stores()

    result = sales_by_store(
        orders,
        stores,
    )

    assert "store_id" in result.columns
    assert "store_name" in result.columns
    assert "region" in result.columns
    assert "revenue" in result.columns

    assert result["revenue"].sum() == pytest.approx(
        15396.70
    )


def test_inventory_stockout_frequency():
    inventory = get_inventory()

    result = calculate_inventory_stockout_frequency(
        inventory
    )

    assert "stockout_frequency" in result.columns

    # P003/S001 has stock 0.
    # P004/S003 has stock -2.
    assert result["stockout_frequency"].max() == pytest.approx(
        1.0
    )