# This modules fetches the ML results from the database and returns to FastAPI Backend
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.session import get_db
from app.models.job import Job, JobStatus
from app.models.insight import Insight
from app.schemas.insight import InsightResponse

router = APIRouter()

@router.get("/{job_id}", response_model = InsightResponse)
async def get_insights(job_id : str, db : AsyncSession = Depends(get_db)):
    """
    Returns ML results and LLM report for a completed job.
    Returns 400 if the pipeline hasnt finished yet.
    Returns 404 if the job does'nt exist.

    👉 This registers a route:
    GET /api/v1/insights/{job_id}
    FastAPI internally stores something like:

    Route:
    path = "/api/v1/insights/{job_id}"
    method = GET
    handler = get_insights
    """

    job_result = await db.execute(select(Job).where(Job.id == job_id)) # fetches the job
    job = job_result.scalar_one_or_none()
    if not job:
        raise HTTPException(status_code = 404, detail = "Job not found.") # if job doesnt exists raises error no job exists
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code = 400,
            detail = f"Pipeline not completed yet. Current Status {job.status.values}" # if job not completed raises exception pipeline not completed
        )
    insight_result = await db.execute(
        select(Insight).where(Insight.job_id == job_id) # if completed fetches insight
    )
    insight = insight_result.scalar_one_or_none()
    if not insight:
        raise HTTPException(
            status_code = 404,
            detail = "Insight not found for this job." # if no insights raises exception of no insoght found
        )
    return insight