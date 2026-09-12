import json
from pathlib import Path

from app.evaluate import main


def test_evaluation_files_can_be_created(
    monkeypatch,
    tmp_path,
):
    input_file = (
        Path("evaluation/cases.json")
        .resolve()
    )

    output_file = (
        tmp_path / "results.json"
    )

    artifacts_dir = (
        tmp_path / "artifacts"
    )

    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate",
            "--input",
            str(input_file),
            "--output",
            str(output_file),
            "--artifacts-dir",
            str(artifacts_dir),
        ],
    )

    main()

    assert output_file.exists()
    assert artifacts_dir.exists()

    data = json.loads(
        output_file.read_text(
            encoding="utf-8"
        )
    )

    assert "cases" in data
    assert "summary" in data
    assert "artifacts" in data

    assert data["summary"]["total"] == 4
