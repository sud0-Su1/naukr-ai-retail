from __future__ import annotations

from typing import Any

import pandas as pd


def calculate_revenue(orders: pd.DataFrame) -> float:
    """
    Calculate total revenue.

    Revenue = quantity * unit_price * (1 - discount)
    """

    revenue = (
        orders["quantity"]
        * orders["unit_price"]
        * (1 - orders["discount"])
    )

    return float(revenue.sum())


def calculate_aov(orders: pd.DataFrame) -> float:
    """
    Calculate Average Order Value.

    AOV = total revenue / number of unique orders
    """

    total_revenue = calculate_revenue(orders)

    unique_orders = orders["order_id"].nunique()

    if unique_orders == 0:
        return 0.0

    return float(total_revenue / unique_orders)


def calculate_return_rate(orders: pd.DataFrame) -> float:
    """
    Calculate return rate based on orders marked as returned.

    Return rate = returned orders / total unique orders
    """

    total_orders = orders["order_id"].nunique()

    if total_orders == 0:
        return 0.0

    returned_orders = orders.loc[
        orders["returned"]
        .astype(str)
        .str.strip()
        .str.lower()
        .isin(["yes", "true", "1"]),
        "order_id",
    ].nunique()

    return float(returned_orders / total_orders)


def sales_by_category(
    orders: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate revenue grouped by normalized product category.
    """

    required_order_columns = {
        "product_id",
        "quantity",
        "unit_price",
        "discount",
    }

    required_product_columns = {
        "product_id",
        "category",
    }

    missing_orders = required_order_columns - set(orders.columns)
    missing_products = required_product_columns - set(products.columns)

    if missing_orders:
        raise ValueError(
            f"Missing order columns: {sorted(missing_orders)}"
        )

    if missing_products:
        raise ValueError(
            f"Missing product columns: {sorted(missing_products)}"
        )

    products_for_join = products.copy()

    if "category_normalized" not in products_for_join.columns:
        from app.retail.normalization import normalize_categories

        products_for_join = normalize_categories(
            products_for_join
        )

    merged = orders.merge(
        products_for_join[
            [
                "product_id",
                "category_normalized",
            ]
        ],
        on="product_id",
        how="left",
        validate="many_to_one",
    )

    merged["revenue"] = (
        merged["quantity"]
        * merged["unit_price"]
        * (1 - merged["discount"])
    )

    result = (
        merged.groupby(
            "category_normalized",
            dropna=False,
        )["revenue"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "category_normalized": "category"
            }
        )
        .sort_values(
            "revenue",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return result


def sales_by_store(
    orders: pd.DataFrame,
    stores: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate revenue grouped by store.
    """

    required_order_columns = {
        "store_id",
        "quantity",
        "unit_price",
        "discount",
    }

    required_store_columns = {
        "store_id",
        "store_name",
        "region",
    }

    missing_orders = required_order_columns - set(orders.columns)
    missing_stores = required_store_columns - set(stores.columns)

    if missing_orders:
        raise ValueError(
            f"Missing order columns: {sorted(missing_orders)}"
        )

    if missing_stores:
        raise ValueError(
            f"Missing store columns: {sorted(missing_stores)}"
        )

    merged = orders.merge(
        stores[["store_id", "store_name", "region"]],
        on="store_id",
        how="left",
        validate="many_to_one",
    )

    merged["revenue"] = (
        merged["quantity"]
        * merged["unit_price"]
        * (1 - merged["discount"])
    )

    result = (
        merged.groupby(
            ["store_id", "store_name", "region"],
            dropna=False,
        )["revenue"]
        .sum()
        .reset_index()
        .sort_values("revenue", ascending=False)
        .reset_index(drop=True)
    )

    return result


def calculate_inventory_stockout_frequency(
    inventory: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate how frequently each product/store combination
    has zero or negative stock.

    Stock-out frequency =
    stockout observations / total observations
    """

    required_columns = {
        "product_id",
        "store_id",
        "stock_qty",
    }

    missing = required_columns - set(inventory.columns)

    if missing:
        raise ValueError(
            f"Missing inventory columns: {sorted(missing)}"
        )

    inventory_copy = inventory.copy()

    inventory_copy["stockout"] = (
        pd.to_numeric(
            inventory_copy["stock_qty"],
            errors="coerce",
        )
        <= 0
    )

    result = (
        inventory_copy.groupby(
            ["product_id", "store_id"],
            dropna=False,
        )
        .agg(
            observations=("stock_qty", "size"),
            stockout_observations=("stockout", "sum"),
        )
        .reset_index()
    )

    result["stockout_frequency"] = (
        result["stockout_observations"]
        / result["observations"]
    )

    return result.sort_values(
        "stockout_frequency",
        ascending=False,
    ).reset_index(drop=True)