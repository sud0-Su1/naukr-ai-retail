from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.chat.planner import MockPlanner
from app.chat.service import ChatService
from app.cleaning.cleaner import clean_orders
from app.cleaning.detector import detect_issues
from app.cleaning.planner import build_cleaning_plan
from app.cleaning.validator import check_idempotency, validate_cleaning


def load_cases(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data

    if isinstance(data, dict) and "cases" in data:
        return data["cases"]

    raise ValueError(
        "Evaluation input must be either a list of cases "
        "or an object containing a 'cases' list."
    )


def evaluate_cleaning_case(case: dict, artifacts_dir: Path) -> dict:
    input_path = Path(case["input"])

    before = __import__("pandas").read_csv(input_path)

    issues = detect_issues(before)
    plan = build_cleaning_plan(issues)

    cleaned, applied_plan = clean_orders(before, plan)

    validation = validate_cleaning(
        before,
        cleaned,
        applied_plan,
    )

    idempotent = check_idempotency(
        clean_orders,
        cleaned,
        applied_plan,
    )

    expected = case.get("expected", {})

    checks = {
        "rows_before": len(before) == expected.get(
            "rows_before",
            len(before),
        ),
        "rows_after": len(cleaned) == expected.get(
            "rows_after",
            len(cleaned),
        ),
        "schema_unchanged": validation["schema"]["unchanged"],
        "idempotent": idempotent["idempotent"],
    }

    case_artifact = artifacts_dir / f"{case['id']}.json"

    artifact = {
        "case_id": case["id"],
        "type": "cleaning",
        "status": "ok" if all(checks.values()) else "failed",
        "checks": checks,
        "issues": issues,
        "cleaning_plan": applied_plan,
        "validation": validation,
    }

    case_artifact.write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )

    return artifact


def evaluate_chat_case(case: dict, artifacts_dir: Path) -> dict:
    """
    Evaluate chat through the same ChatService pipeline used by the app.

    We inject MockPlanner only to keep evaluation deterministic and offline.
    """

    service = ChatService(
        planner=MockPlanner(),
        dataset_path="data/cleaned/canonical_retail.csv",
    )

    response = service.answer(case["question"])

    expected = case.get("expected", {})

    checks = {}

    expected_status = expected.get("status")

    if expected_status is not None:
        checks["status"] = response["status"] == expected_status

    # ---------------------------------------------------------
    # Successful response checks
    # ---------------------------------------------------------
    if response["status"] == "ok":
        result = response.get("result", [])

        if "top_category" in expected:
            if result:
                actual_category = result[0].get("category_normalized")
                checks["top_category"] = (
                    actual_category == expected["top_category"]
                )
            else:
                checks["top_category"] = False

        if "metric_alias" in expected:
            metric_alias = expected["metric_alias"]

            checks["metric_present"] = any(
                metric_alias in row
                for row in result
            )

        if expected.get("grounded") is True:
            evidence = response.get("evidence", {})
            checks["grounded"] = bool(
                evidence
                and "execution_trace" in evidence
                and "rows_considered" in evidence
            )

    # ---------------------------------------------------------
    # Expected planner failure / refusal
    # ---------------------------------------------------------
    if expected.get("status") == "error":
        checks["error_present"] = response["status"] == "error"

    case_status = "ok" if all(checks.values()) else "failed"

    artifact = {
        "case_id": case["id"],
        "type": "chat",
        "status": case_status,
        "question": case["question"],
        "checks": checks,
        "response": response,
    }

    case_artifact = artifacts_dir / f"{case['id']}.json"

    case_artifact.write_text(
        json.dumps(artifact, indent=2),
        encoding="utf-8",
    )

    return artifact


def evaluate_case(case: dict, artifacts_dir: Path) -> dict:
    try:
        if case["type"] == "cleaning":
            return evaluate_cleaning_case(
                case,
                artifacts_dir,
            )

        if case["type"] == "chat":
            return evaluate_chat_case(
                case,
                artifacts_dir,
            )

        return {
            "case_id": case["id"],
            "status": "failed",
            "error_type": "unknown_case_type",
            "message": case["type"],
        }

    except Exception as exc:
        return {
            "case_id": case["id"],
            "status": "failed",
            "error_type": type(exc).__name__,
            "message": str(exc),
        }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    parser.add_argument(
        "--artifacts-dir",
        required=True,
    )

    args = parser.parse_args()

    cases = load_cases(args.input)

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = []

    for case in cases:
        result = evaluate_case(
            case,
            artifacts_dir,
        )

        results.append(result)

    passed = sum(
        1
        for result in results
        if result["status"] == "ok"
    )

    failed = len(results) - passed

    summary = {
        "total": len(results),
        "passed": passed,
        "failed": failed,
    }

    artifact_files = [
        str(path)
        for path in sorted(artifacts_dir.glob("*.json"))
    ]

    output = {
        "summary": summary,
        "cases": results,
        "artifacts": {
            "directory": str(artifacts_dir),
            "files": artifact_files,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "total": len(results),
                "passed": passed,
                "failed": failed,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()