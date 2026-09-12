import json

import pandas as pd

from app.cleaning.detector import detect_issues


if __name__ == "__main__":
    df = pd.read_csv("data/sample/orders.csv")

    issues = detect_issues(df)

    print(
        json.dumps(
            issues,
            indent=2,
            ensure_ascii=False,
        )
    )
