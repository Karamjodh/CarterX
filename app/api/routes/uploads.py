import io
import pandas as pd
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.db.session import get_db
from app.models.job import Job, JobStatus
from app.schemas.job import JobResponse
from app.core.config import Settings

settings = Settings()
router = APIRouter()
@router.post("/", response_model = JobResponse, status_code = 202)
async def upload_file(
    file: UploadFile = File(...),
    db : AsyncSession = Depends(get_db)
):
    import os
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
    except Exception as e:
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
            "forecasting" : "pending",
            "llm_report" : "pending",
        }
    )
    db.add(job)
    await db.flush()
    return job
