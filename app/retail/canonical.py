from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.retail.joins import safe_join
from app.retail.normalization import normalize_categories


ORDERS_PATH = Path("data/cleaned/orders_clean.csv")
PRODUCTS_PATH = Path("data/sample/products.csv")
CUSTOMERS_PATH = Path("data/sample/customers.csv")
STORES_PATH = Path("data/sample/stores.csv")
OUTPUT_PATH = Path("data/cleaned/canonical_retail.csv")


def build_canonical_retail() -> tuple[pd.DataFrame, dict]:
    """
    Build the canonical retail analytical dataset.

    The resulting dataset contains:
    - cleaned order data
    - normalized product category
    - customer region
    - store region
    - deterministic revenue
    """

    orders = pd.read_csv(ORDERS_PATH)
    products = pd.read_csv(PRODUCTS_PATH)
    customers = pd.read_csv(CUSTOMERS_PATH)
    stores = pd.read_csv(STORES_PATH)

    # Normalize product categories first.
    products = normalize_categories(products)

    # orders -> products
    joined, products_audit = safe_join(
        orders,
        products[
            [
                "product_id",
                "sku",
                "product_name",
                "category_normalized",
                "brand",
                "price",
                "cost",
            ]
        ],
        left_key="product_id",
        right_key="product_id",
        relationship="many_to_one",
    )

    # orders/products -> customers
    joined, customers_audit = safe_join(
        joined,
        customers,
        left_key="customer_id",
        right_key="customer_id",
        relationship="many_to_one",
        rename_right={
            "region": "customer_region",
        },
    )

    # previous result -> stores
    joined, stores_audit = safe_join(
        joined,
        stores,
        left_key="store_id",
        right_key="store_id",
        relationship="many_to_one",
        rename_right={
            "region": "store_region",
        },
    )

    # Deterministic business metric.
    joined["revenue"] = (
        joined["quantity"]
        * joined["unit_price"]
        * (1 - joined["discount"])
    )

    # Keep a predictable analytical schema.
    columns = [
        "order_id",
        "customer_id",
        "product_id",
        "store_id",
        "order_date",
        "quantity",
        "unit_price",
        "discount",
        "returned",
        "sku",
        "product_name",
        "category_normalized",
        "brand",
        "price",
        "cost",
        "name",
        "customer_region",
        "signup_date",
        "store_name",
        "store_region",
        "city",
        "revenue",
    ]

    canonical = joined[columns].copy()

    audit = {
        "products_join": products_audit,
        "customers_join": customers_audit,
        "stores_join": stores_audit,
        "rows": len(canonical),
        "columns": list(canonical.columns),
    }

    return canonical, audit