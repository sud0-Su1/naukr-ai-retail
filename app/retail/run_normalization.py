import pandas as pd

from app.retail.normalization import normalize_categories


if __name__ == "__main__":
    products = pd.read_csv(
        "data/sample/products.csv"
    )

    normalized = normalize_categories(products)

    print(
        normalized[
            [
                "product_id",
                "category_original",
                "category_normalized",
                "category_normalization_method",
                "category_confidence",
            ]
        ].to_string(index=False)
    )