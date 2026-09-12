import json

import pandas as pd

from app.cleaning.detector import detect_issues
from app.cleaning.planner import build_cleaning_plan


if __name__ == "__main__":
    df = pd.read_csv("data/sample/orders.csv")

    issues = detect_issues(df)
    plan = build_cleaning_plan(issues)

    print(
        json.dumps(
            plan,
            indent=2,
            ensure_ascii=False,
        )
    )