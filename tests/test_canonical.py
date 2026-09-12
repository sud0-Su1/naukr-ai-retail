from pathlib import Path

from app.retail.canonical import build_canonical_retail


def test_canonical_dataset():
    canonical, audit = build_canonical_retail()

    assert len(canonical) == 5

    required_columns = {
        "order_id",
        "product_id",
        "store_id",
        "category_normalized",
        "customer_region",
        "store_region",
        "revenue",
    }

    assert required_columns.issubset(
        set(canonical.columns)
    )

    assert canonical["revenue"].notna().all()

    assert canonical["category_normalized"].notna().all()

    assert audit["products_join"]["safe"] is True
    assert audit["customers_join"]["safe"] is True
    assert audit["stores_join"]["safe"] is True

    assert audit["products_join"]["fanout_ratio"] == 1.0
    assert audit["customers_join"]["fanout_ratio"] == 1.0
    assert audit["stores_join"]["fanout_ratio"] == 1.0