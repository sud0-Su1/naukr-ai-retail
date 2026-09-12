import json

from app.retail.canonical import (
    OUTPUT_PATH,
    build_canonical_retail,
)


if __name__ == "__main__":
    canonical, audit = build_canonical_retail()

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    canonical.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("Canonical retail dataset created.")
    print(f"Rows: {len(canonical)}")
    print(f"Output: {OUTPUT_PATH}")

    print("\nAudit:")
    print(
        json.dumps(
            audit,
            indent=2,
            ensure_ascii=False,
        )
    )

    print("\nPreview:")
    print(canonical.head().to_string(index=False))