import json

import pandas as pd

from app.retail.joins import safe_join


if __name__ == "__main__":

    orders = pd.read_csv(
        "data/cleaned/orders_clean.csv"
    )

    products = pd.read_csv(
        "data/sample/products.csv"
    )

    customers = pd.read_csv(
        "data/sample/customers.csv"
    )

    stores = pd.read_csv(
        "data/sample/stores.csv"
    )

    products_joined, products_audit = safe_join(
        orders,
        products,
        left_key="product_id",
        right_key="product_id",
        relationship="many_to_one",
    )

    customers_joined, customers_audit = safe_join(
    products_joined,
    customers,
    left_key="customer_id",
    right_key="customer_id",
    relationship="many_to_one",
    rename_right={
        "region": "customer_region",
    },
)

    final_dataset, stores_audit = safe_join(
    customers_joined,
    stores,
    left_key="store_id",
    right_key="store_id",
    relationship="many_to_one",
    rename_right={
        "region": "store_region",
    },
)
    result = {
        "products_join": products_audit,
        "customers_join": customers_audit,
        "stores_join": stores_audit,
        "final_rows": len(final_dataset),
        "final_columns": list(final_dataset.columns),
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )