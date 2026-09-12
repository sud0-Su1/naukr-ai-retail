import json
from pathlib import Path

import pandas as pd

from app.cleaning.cleaner import clean_orders
from app.cleaning.detector import detect_issues
from app.cleaning.planner import build_cleaning_plan


INPUT_PATH = Path("data/sample/orders.csv")
OUTPUT_PATH = Path("data/cleaned/orders_clean.csv")


if __name__ == "__main__":
    df = pd.read_csv(INPUT_PATH)

    issues = detect_issues(df)

    plan = build_cleaning_plan(issues)

    cleaned, updated_plan = clean_orders(
        df=df,
        plan=plan,
    )

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cleaned.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("Cleaning completed.")
    print(f"Input rows:   {len(df)}")
    print(f"Output rows:  {len(cleaned)}")
    print(f"Output file:  {OUTPUT_PATH}")

    print("\nUpdated cleaning plan:")
    print(
        json.dumps(
            updated_plan,
            indent=2,
            ensure_ascii=False,
        )
    )