"""
CarterX — Geo Analysis Route
GET /api/v1/geo/{job_id}
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from app.db.session import get_db
from app.models.job import Job, JobStatus
from app.models.insight import Insight

router = APIRouter()


@router.get("/{job_id}")
async def get_geo(job_id: str, db: AsyncSession = Depends(get_db)):
    """
    Returns geographic analysis data for a completed job.
    GeoTab.js reads from insights.geo_data directly,
    but this route allows direct API access.
    """
    job_result = await db.execute(select(Job).where(Job.id == job_id))
    job        = job_result.scalar_one_or_none()

    if not job:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found.")

    if job.status != JobStatus.COMPLETED:
        raise HTTPException(
            status_code=400,
            detail=f"Job not completed yet. Current status: {job.status.value}"
        )

    insight_result = await db.execute(
        select(Insight).where(Insight.job_id == job_id)
    )
    insight = insight_result.scalar_one_or_none()

    if not insight:
        raise HTTPException(status_code=404, detail="No insight found for this job.")

    return {
        "job_id":       job_id,
        "dataset_type": insight.dataset_type,
        "geo_data":     insight.geo_data or {"has_geo_data": False},
    }