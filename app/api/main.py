from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.chat.service import ChatService
from app.cleaning.cleaner import clean_orders
from app.cleaning.detector import detect_issues
from app.cleaning.planner import build_cleaning_plan
from app.cleaning.validator import validate_cleaning
from app.profiling.profiler import profile_dataframe
from app.retail.canonical import build_canonical_retail


app = FastAPI(
    title="Naukr.AI Retail Intelligence API",
    version="1.0.0",
    description=(
        "AI Retail Data & Conversational Intelligence Workbench"
    ),
)

DATA_RAW = Path("data/raw")
DATA_CLEANED = Path("data/cleaned")
ARTIFACTS = Path("artifacts/api_runs")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB

ALLOWED_SUFFIX = ".csv"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_CLEANED.mkdir(parents=True, exist_ok=True)
ARTIFACTS.mkdir(parents=True, exist_ok=True)


def safe_filename(filename: str | None) -> str:
    """
    Validate and sanitize an uploaded filename.

    User-controlled filenames must never become arbitrary filesystem paths.
    """
    if not filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required.",
        )

    name = Path(filename).name

    if name != filename:
        raise HTTPException(
            status_code=400,
            detail="Invalid filename.",
        )

    if not name.lower().endswith(ALLOWED_SUFFIX):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are supported.",
        )

    if not re.fullmatch(r"[A-Za-z0-9._-]+", name):
        raise HTTPException(
            status_code=400,
            detail="Filename contains unsupported characters.",
        )

    return name


async def save_upload(upload: UploadFile) -> tuple[Path, str]:
    """
    Save a bounded CSV upload into an isolated run directory.
    """
    filename = safe_filename(upload.filename)

    run_id = uuid.uuid4().hex
    run_dir = DATA_RAW / run_id
    run_dir.mkdir(parents=True, exist_ok=False)

    target = run_dir / filename

    content = await upload.read()

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 10 MB size limit.",
        )

    if not content:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is empty.",
        )

    target.write_bytes(content)

    return target, run_id


def write_artifact(run_id: str, name: str, payload: dict) -> str:
    """
    Persist a machine-readable API artifact.
    """
    run_dir = ARTIFACTS / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    path = run_dir / name

    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )

    return str(path)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "service": "naukr-ai-retail",
        "version": "1.0.0",
    }


@app.post("/profile")
async def profile_upload(
    file: UploadFile = File(...),
) -> dict:
    """
    Upload a CSV and generate a non-mutating data-quality profile.
    """
    path, run_id = await save_upload(file)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read CSV: {exc}",
        ) from exc

    profile = profile_dataframe(
        df,
        dataset_name=path.name,
    )

    artifact = {
        "run_id": run_id,
        "original_filename": path.name,
        "profile": profile,
    }

    artifact_path = write_artifact(
        run_id,
        "profile.json",
        artifact,
    )

    return {
        **artifact,
        "artifact": artifact_path,
    }


@app.post("/clean")
async def clean_upload(
    file: UploadFile = File(...),
) -> dict:
    """
    Execute the deterministic cleaning workflow:

    upload
      -> profile
      -> detect
      -> cleaning plan
      -> clean
      -> validate
      -> persist cleaned artifact
    """
    path, run_id = await save_upload(file)

    try:
        before = pd.read_csv(path)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unable to read CSV: {exc}",
        ) from exc

    profile_before = profile_dataframe(
        before,
        dataset_name=path.name,
    )

    issues = detect_issues(before)

    plan = build_cleaning_plan(issues)

    try:
        cleaned, applied_plan = clean_orders(
            before,
            plan,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Cleaning failed: {exc}",
        ) from exc

    profile_after = profile_dataframe(
        cleaned,
        dataset_name=f"{path.stem}_clean.csv",
    )

    validation = validate_cleaning(
        before,
        cleaned,
        unresolved_issues=[],
    )

    cleaned_dir = DATA_CLEANED / run_id
    cleaned_dir.mkdir(parents=True, exist_ok=True)

    cleaned_path = cleaned_dir / f"{path.stem}_clean.csv"
    cleaned.to_csv(cleaned_path, index=False)

    artifact = {
        "run_id": run_id,
        "original_filename": path.name,
        "profile_before": profile_before,
        "issues": issues,
        "cleaning_plan": applied_plan,
        "profile_after": profile_after,
        "validation": validation,
        "cleaned_artifact": str(cleaned_path),
    }

    artifact_path = write_artifact(
        run_id,
        "cleaning_run.json",
        artifact,
    )

    return {
        **artifact,
        "artifact": artifact_path,
    }


@app.post("/chat")
def chat(
    question: str = Form(...),
    session_id: str = Form("default-session"),
) -> dict:
    """
    Execute one turn of the persistent retail chat workflow.

    The ChatService is responsible for:
      LLM/mock planner
        -> structured plan
        -> plan validation
        -> deterministic execution
        -> evidence
        -> session persistence
    """
    if not question.strip():
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", session_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid session_id.",
        )

    service = ChatService(
        session_id=session_id,
    )

    return service.answer(question)


@app.post("/retail/canonical")
def build_canonical() -> dict:
    """
    Rebuild the canonical retail analytical dataset from the
    cleaned orders and reference datasets.
    """
    try:
        canonical, audit = build_canonical_retail()
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Canonical dataset build failed: {exc}",
        ) from exc

    output_path = DATA_CLEANED / "canonical_retail.csv"

    canonical.to_csv(
        output_path,
        index=False,
    )

    return {
        "status": "ok",
        "rows": len(canonical),
        "columns": list(canonical.columns),
        "audit": audit,
        "artifact": str(output_path),
    }
