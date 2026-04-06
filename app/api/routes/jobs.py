from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.models.job import Job, JobStatus
from app.schemas.job import JobCreate, JobResponse

router = APIRouter()

@router.post("/", response_model = JobResponse, status_code = 201)
async def create_job(data : JobCreate, db : AsyncSession = Depends(get_db)):
    new_job = Job(
        filename = data.filename,
        row_count = data.row_count,
        status = JobStatus.PENDING,
        stage_status = {
            "preprocessing" : "pending",
            "segmentation" : "pending",
            "association_rules" : "pending",
            "forecasting" : "pending",
            "llm_report" : "pending"
        }
    )
    db.add(new_job)
    await db.flush()
    return new_job

@router.get("/{job_id}", response_model = JobResponse)
async def get_job(job_id : str, db : AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).where(Job.id == job_id))
    job = result.scalar_one_or_none()
    if job is None:
        raise HTTPException(status_code = 404, detail = f"Job {job_id} not found")
    return job

@router.get("/", response_model = list[JobResponse])
async def get_all_jobs(db : AsyncSession = Depends(get_db)):
    result = await db.execute(select(Job).order_by(Job.created_at.desc()))
    jobs = result.scalars().all()
    return jobs