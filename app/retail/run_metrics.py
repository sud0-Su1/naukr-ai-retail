import json

import pandas as pd

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


if __name__ == "__main__":
    orders = pd.read_csv(
        "data/sample/orders.csv"
    )

    products = pd.read_csv(
        "data/sample/products.csv"
    )

    stores = pd.read_csv(
        "data/sample/stores.csv"
    )

    inventory = pd.read_csv(
        "data/sample/inventory.csv"
    )

    # Clean orders first.
    issues = detect_issues(orders)
    plan = build_cleaning_plan(issues)

    clean_orders_df, _ = clean_orders(
        orders,
        plan,
    )

    result = {
        "revenue": calculate_revenue(
            clean_orders_df
        ),
        "aov": calculate_aov(
            clean_orders_df
        ),
        "return_rate": calculate_return_rate(
            clean_orders_df
        ),
        "sales_by_category": (
            sales_by_category(
                clean_orders_df,
                products,
            ).to_dict(orient="records")
        ),
        "sales_by_store": (
            sales_by_store(
                clean_orders_df,
                stores,
            ).to_dict(orient="records")
        ),
        "stockout_frequency": (
            calculate_inventory_stockout_frequency(
                inventory
            ).to_dict(orient="records")
        ),
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )