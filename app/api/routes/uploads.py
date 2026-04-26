import io
import os
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db, AsyncSessionLocal
from app.models.job import Job, JobStatus
from app.schemas.job import JobResponse
from app.core.config import Settings
from app.services.pipeline import run_pipeline

settings = Settings()
router = APIRouter()
@router.post("/", response_model = JobResponse, status_code = 202)
async def upload_file(
    background_tasks : BackgroundTasks,
    file: UploadFile = File(...),
    db : AsyncSession = Depends(get_db),
):
    """
    Accepts a CSV/XLSX file, creates a job, and starts
    the ML pipeline in the background.

    Returns immediately with the job_id.
    Frontend polls GET /api/v1/jobs/{job_id} for progress.
    """

    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code = 400,
            detail = f"Invalid file type '{file_ext}'. Only CSV and XLSX are allowed."
        )
    contents = await file.read()
    size_mb = len(contents)/(1024*1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code = 400,
            detail = f"File size exceeds the maximum limit of {settings.MAX_UPLOAD_SIZE_MB} MB."
        )
    try:
        if file_ext == ".csv":
            df = pd.read_csv(io.BytesIO(contents))
        else:
            df = pd.read_excel(io.BytesIO(contents))
    except Exception:
        raise HTTPException(
            status_code = 400,
            detail = f"Could not read file. Make sure it is a valid CSV or XLSX."
        )
    if len(df) < settings.MIN_ROWS_REQUIRED:
        raise HTTPException(
            status_code = 400,
            detail = f"File must contain at least {settings.MIN_ROWS_REQUIRED} rows."
        )
    job = Job(
        filename = file.filename,
        row_count = len(df),
        status = JobStatus.PENDING,
        stage_status = {
            "preprocessing" : "pending",
            "segmentation" : "pending",
            "association_rules" : "pending",
            "llm_report" : "pending",
        }
    )
    db.add(job)
    await db.flush()

    job_id = job.id
    content_type  = file.content_type
    background_tasks.add_task(
        _run_pipeline_task,
        job_id = job_id,
        file_bytes = contents,
        content_type = content_type
    )
    return job

async def _run_pipeline_task(
    job_id:       str,
    file_bytes:   bytes,
    content_type: str,
):
    """
    Wrapper that creates a fresh database session for the background task.

    Important: background tasks run outside the request lifecycle.
    The db session from get_db() closes when the request ends —
    so we need a new session specifically for the background work.
    """
    async with AsyncSessionLocal() as db:
        await run_pipeline(
            job_id       = job_id,
            file_bytes   = file_bytes,
            content_type = content_type,
            db           = db,
        )