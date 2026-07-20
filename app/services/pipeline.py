import uuid
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update
from app.models.job import Job, JobStatus
from app.models.insight import Insight
from app.services.ml.preprocessing import run_preprocessing
from app.services.ml.segmentation import run_segmentation
from app.services.ml.association_rules import run_association_rules
from app.services.ml.forecasting import run_forecasting
from app.services.ml.geo_analysis import run_geo_analysis
from app.services.prompt_builder import build_analysis_prompt
from app.services.llm import generate_report
from app.services.ml.tsne import run_tsne
import json

logger = logging.getLogger(__name__)


async def run_pipeline(
    job_id:       str,
    file_bytes:   bytes,
    content_type: str,
    db:           AsyncSession,
    llm_model:    str = "groq",
    focus:        str = "general",
):
    async def update_stage(stage: str, status: str):
        """Updates a single stage status using a direct SQL UPDATE."""
        result = await db.execute(select(Job).where(Job.id == job_id))
        job    = result.scalar_one_or_none()
        if job:
            stages           = dict(job.stage_status or {})
            stages[stage]    = status
            await db.execute(
                update(Job)
                .where(Job.id == job_id)
                .values(stage_status=stages)
            )
            await db.commit()
            logger.info(f"Job {job_id} — [{stage}] {status}")

    async def set_job_status(status: JobStatus, error: str = None):
        """Sets top-level job status using a direct SQL UPDATE — avoids ORM dirty-tracking issues."""
        values = {"status": status}
        if error:
            values["error_message"] = error
        await db.execute(
            update(Job)
            .where(Job.id == job_id)
            .values(**values)
        )
        await db.commit()
        logger.info(f"Job {job_id} — status set to {status}")

    try:
        await set_job_status(JobStatus.PROCESSING)

        # ── Stage 1: Preprocessing ─────────────────────────────────────────
        await update_stage("preprocessing", "running")
        prep = run_preprocessing(file_bytes, content_type)
        await update_stage("preprocessing", "completed")

        # ── Stage 2: Segmentation ──────────────────────────────────────────
        await update_stage("segmentation", "running")
        seg = run_segmentation(prep.df_rfm)
        await update_stage("segmentation", "completed")

        # ── Stage 3: t-SNE ────────────────────────────────────────────────
        await update_stage("tsne", "running")
        logger.info(f"[{job_id}] Running t-SNE...")
        tsne_result          = run_tsne(seg.df_rfm_labelled, seg.cluster_profiles)
        tsne_data_serialized = json.loads(json.dumps(tsne_result.embedding_2d))
        await update_stage("tsne", "completed")

        # ── Stage 4: Association rules ─────────────────────────────────────
        await update_stage("association_rules", "running")
        assoc = run_association_rules(prep.df_basket)
        await update_stage("association_rules", "completed")

        # ── Stage 5: Forecasting ───────────────────────────────────────────
        await update_stage("forecasting", "running")
        logger.info(f"[{job_id}] Running forecasting...")
        forecast_result          = run_forecasting(prep.df_clean, prep.dataset_type)
        forecast_data_serialized = json.loads(json.dumps({
            "success":        forecast_result.success,
            "model_used":     forecast_result.model_used,
            "history":        forecast_result.history,
            "forecast":       forecast_result.forecast,
            "horizons":       forecast_result.horizons,
            "mae":            forecast_result.mae,
            "has_date_data":  forecast_result.has_date_data,
            "warning":        forecast_result.warning,
            "error":          forecast_result.error,
        }))
        await update_stage("forecasting", "completed")

        # ── Stage 6: Geo Analysis ────────────────────────────────────────────
        await update_stage("geo_analysis", "running")
        geo_result = run_geo_analysis(prep.df_clean, seg.cluster_profiles, prep.dataset_type)
        geo_data_serialized = json.loads(json.dumps({
            "has_geo_data":         geo_result.has_geo_data,
            "geo_column":           geo_result.geo_column,
            "region_stats":         geo_result.region_stats,
            "region_growth":        geo_result.region_growth,
            "top_regions":          geo_result.top_regions,
            "regional_segments":    geo_result.regional_segments,
            "regional_products":    geo_result.regional_products,
            "market_concentration": geo_result.market_concentration,
            "summary":              geo_result.summary,
        }))
        await update_stage("geo_analysis", "completed")

        # ── Stage 7: LLM report ────────────────────────────────────────────
        await update_stage("llm_report", "running")
        analysis_data = {
            "summary":           prep.summary,
            "segments":          seg.cluster_profiles,
            "association_rules": assoc.rules,
            "trend_data":        prep.trend_data,
            "silhouette_score":  seg.silhouette_score,
            "dataset_type":      prep.dataset_type,
        }
        prompt     = build_analysis_prompt(analysis_data, focus=focus)
        llm_result = await generate_report(prompt, model=llm_model)
        await update_stage("llm_report", "completed")

        trend_data_serialized = json.loads(json.dumps(prep.trend_data)) if prep.trend_data else None

        # ── Persist insight ────────────────────────────────────────────────
        insight = Insight(
            id                = str(uuid.uuid4()),
            job_id            = job_id,
            summary           = prep.summary,
            cluster_profiles  = seg.cluster_profiles,
            association_rules = assoc.rules,
            n_clusters        = seg.n_clusters,
            silhouette_score  = seg.silhouette_score,
            trend_data        = trend_data_serialized,
            tsne_data         = tsne_data_serialized,
            forecast_data     = forecast_data_serialized,
            geo_data          = geo_data_serialized,
            llm_report        = llm_result["text"],
            model_used        = llm_result["model_used"],
            dataset_type      = prep.dataset_type,
        )
        db.add(insight)
        await db.commit()
        await set_job_status(JobStatus.COMPLETED)
        logger.info(f"Job {job_id} completed successfully")

        # ── Mark job completed AFTER insight is saved ──────────────────────
        await set_job_status(JobStatus.COMPLETED)
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}", exc_info=True)
        await set_job_status(JobStatus.FAILED, error=str(e))
        raise