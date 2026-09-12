from pathlib import Path

import pandas as pd
import pytest

from app.retail.joins import (
    UnsafeJoinError,
    safe_join,
)


BASE_DIR = Path(__file__).resolve().parents[1]


def load_orders() -> pd.DataFrame:
    return pd.read_csv(
        BASE_DIR / "data" / "sample" / "orders.csv"
    )


def load_products() -> pd.DataFrame:
    return pd.read_csv(
        BASE_DIR / "data" / "sample" / "products.csv"
    )


def load_customers() -> pd.DataFrame:
    return pd.read_csv(
        BASE_DIR / "data" / "sample" / "customers.csv"
    )


def load_stores() -> pd.DataFrame:
    return pd.read_csv(
        BASE_DIR / "data" / "sample" / "stores.csv"
    )


def test_orders_products_many_to_one():
    orders = load_orders()
    products = load_products()

    joined, audit = safe_join(
        orders,
        products,
        left_key="product_id",
        right_key="product_id",
        relationship="many_to_one",
    )

    assert audit["safe"] is True
    assert audit["rows_after"] == len(orders)
    assert audit["fanout_ratio"] == pytest.approx(1.0)

    assert audit["unmatched_left_rows"] == 0


def test_orders_customers_many_to_one():
    orders = load_orders()
    customers = load_customers()

    joined, audit = safe_join(
        orders,
        customers,
        left_key="customer_id",
        right_key="customer_id",
        relationship="many_to_one",
    )

    assert audit["safe"] is True
    assert audit["rows_after"] == len(orders)
    assert audit["unmatched_left_rows"] == 0


def test_orders_stores_many_to_one():
    orders = load_orders()
    stores = load_stores()

    joined, audit = safe_join(
        orders,
        stores,
        left_key="store_id",
        right_key="store_id",
        relationship="many_to_one",
    )

    assert audit["safe"] is True
    assert audit["rows_after"] == len(orders)
    assert audit["unmatched_left_rows"] == 0


def test_duplicate_dimension_key_is_rejected():
    orders = load_orders()
    products = load_products()

    # Deliberately create a bad dimension table.
    products = pd.concat(
        [
            products,
            products.iloc[[0]],
        ],
        ignore_index=True,
    )

    with pytest.raises(UnsafeJoinError):
        safe_join(
            orders,
            products,
            left_key="product_id",
            right_key="product_id",
            relationship="many_to_one",
        )


def test_unknown_join_key_is_rejected():
    orders = load_orders()
    products = load_products()

    with pytest.raises(UnsafeJoinError):
        safe_join(
            orders,
            products,
            left_key="does_not_exist",
            right_key="product_id",
            relationship="many_to_one",
        )

def test_region_columns_are_explicit():
    orders = load_orders().drop_duplicates()
    customers = load_customers()
    stores = load_stores()

    joined, _ = safe_join(
        orders,
        customers,
        left_key="customer_id",
        right_key="customer_id",
        relationship="many_to_one",
        rename_right={
            "region": "customer_region",
        },
    )

    joined, _ = safe_join(
        joined,
        stores,
        left_key="store_id",
        right_key="store_id",
        relationship="many_to_one",
        rename_right={
            "region": "store_region",
        },
    )

    assert "customer_region" in joined.columns
    assert "store_region" in joined.columns
    assert "region_x" not in joined.columns
    assert "region_y" not in joined.columns