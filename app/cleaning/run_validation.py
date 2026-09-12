import json

import pandas as pd

from app.cleaning.cleaner import clean_orders
from app.cleaning.detector import detect_issues
from app.cleaning.planner import build_cleaning_plan
from app.cleaning.validator import (
    check_idempotency,
    validate_cleaning,
)


if __name__ == "__main__":
    input_path = "data/sample/orders.csv"

    before = pd.read_csv(input_path)

    # 1. Detect issues
    issues = detect_issues(before)

    # 2. Build cleaning plan
    plan = build_cleaning_plan(issues)

    # 3. Clean
    cleaned, updated_plan = clean_orders(
        before,
        plan,
    )

    # 4. Validate
    validation = validate_cleaning(
        before,
        cleaned,
        unresolved_issues=[],
    )

    # 5. Check idempotency
    idempotency = check_idempotency(
        clean_orders,
        cleaned,
        updated_plan,
    )

    result = {
        "profile_before": {
            "rows": len(before),
            "columns": len(before.columns),
        },
        "profile_after": {
            "rows": len(cleaned),
            "columns": len(cleaned.columns),
        },
        "cleaning_plan": updated_plan,
        "validation": validation,
        "idempotency": idempotency,
    }

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False,
        )
    )