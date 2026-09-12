from pathlib import Path

import pandas as pd

from app.retail.normalization import (
    normalize_categories,
    normalize_category_value,
)


BASE_DIR = Path(__file__).resolve().parents[1]


def test_tee_normalization():
    result = normalize_category_value("tee")

    assert result["original"] == "tee"
    assert result["normalized"] == "T-Shirt"
    assert result["method"] == "alias_dictionary"
    assert result["confidence"] == 1.0


def test_whitespace_normalization():
    result = normalize_category_value(" footwear ")

    assert result["original"] == " footwear "
    assert result["normalized"] == "Footwear"


def test_product_normalization_preserves_original():
    products = pd.read_csv(
        BASE_DIR / "data" / "sample" / "products.csv"
    )

    normalized = normalize_categories(products)

    assert "category_original" in normalized.columns
    assert "category_normalized" in normalized.columns
    assert "category_normalization_method" in normalized.columns
    assert "category_confidence" in normalized.columns

    # Original data is preserved.
    assert normalized.loc[
        normalized["product_id"] == "P003",
        "category_original",
    ].iloc[0] == "tee"

    # tee -> T-Shirt
    assert normalized.loc[
        normalized["product_id"] == "P003",
        "category_normalized",
    ].iloc[0] == "T-Shirt"