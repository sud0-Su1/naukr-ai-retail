from __future__ import annotations

from typing import Any

import pandas as pd


CATEGORY_ALIASES: dict[str, tuple[str, float]] = {
    "tee": ("T-Shirt", 1.0),
    "t shirt": ("T-Shirt", 1.0),
    "t-shirt": ("T-Shirt", 1.0),
    "footwear": ("Footwear", 1.0),
    "shoes": ("Footwear", 0.9),
}


def normalize_category_value(value: object) -> dict[str, Any]:
    """
    Normalize a category while preserving the original value
    and the normalization decision.
    """

    if pd.isna(value):
        return {
            "original": None,
            "normalized": None,
            "method": "missing",
            "confidence": None,
        }

    original = str(value)
    cleaned = original.strip().lower()

    if cleaned in CATEGORY_ALIASES:
        normalized, confidence = CATEGORY_ALIASES[cleaned]

        return {
            "original": original,
            "normalized": normalized,
            "method": "alias_dictionary",
            "confidence": confidence,
        }

    # Known category, only formatting differs.
    return {
        "original": original,
        "normalized": original.strip(),
        "method": "whitespace_normalization",
        "confidence": 1.0,
    }


def normalize_categories(
    products: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add auditable category-normalization columns.

    Original category values are never overwritten.
    """

    if "category" not in products.columns:
        raise ValueError(
            "Products dataset must contain a 'category' column."
        )

    result = products.copy(deep=True)

    normalized = result["category"].map(
        normalize_category_value
    )

    result["category_original"] = normalized.map(
        lambda x: x["original"]
    )

    result["category_normalized"] = normalized.map(
        lambda x: x["normalized"]
    )

    result["category_normalization_method"] = normalized.map(
        lambda x: x["method"]
    )

    result["category_confidence"] = normalized.map(
        lambda x: x["confidence"]
    )

    return result