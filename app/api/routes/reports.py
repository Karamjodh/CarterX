from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.db.session import get_db
from app.models.insight import Insight
from app.schemas.report import ReportRequest, ReportResponse
from app.services.llm import generate_report
from app.services.prompt_builder import build_analysis_prompt

router = APIRouter()


@router.post("/analyze", response_model=ReportResponse)
async def analyze(request: ReportRequest, db: AsyncSession = Depends(get_db)):
    """
    Regenerate a strategy report with a different model or focus.
    Fetches existing ML results from DB — no pipeline re-run needed.
    """
    data = await _get_data(request, db)

    prompt = build_analysis_prompt(data, focus=request.focus.value)

    try:
        result = await generate_report(prompt, model=request.model.value)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {str(e)}")

    return ReportResponse(
        job_id        = request.job_id,
        model_used    = result["model_used"],
        focus         = request.focus.value,
        report        = result["text"],
        input_tokens  = result["input_tokens"],
        output_tokens = result["output_tokens"],
    )


async def _get_data(request: ReportRequest, db: AsyncSession) -> dict:
    # Option A — raw data sent directly (for testing)
    if request.data:
        return request.data

    # Option B — fetch from DB using job_id (production path)
    if request.job_id:
        result  = await db.execute(
            select(Insight).where(Insight.job_id == request.job_id)
        )
        insight = result.scalar_one_or_none()
        if not insight:
            raise HTTPException(
                status_code=404,
                detail=f"No insights found for job {request.job_id}. Run the pipeline first."
            )
        return {
            "summary":           insight.summary,
            "segments":          insight.cluster_profiles,
            "association_rules": insight.association_rules,
            "trend_data":        insight.trend_data,
            "silhouette_score":  insight.silhouette_score,
        }

    raise HTTPException(
        status_code=400,
        detail="Either 'job_id' or 'data' must be provided."
    )