import uuid
import logging
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.job import Job, JobStatus
from app.models.insight import Insight
from app.services.ml.preprocessing import run_preprocessing
from app.services.ml.segmentation import run_segmentation
from app.services.ml.association_rules import run_association_rules
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
        result = await db.execute(select(Job).where(Job.id == job_id))
        job    = result.scalar_one_or_none()
        if job:
            stages           = dict(job.stage_status or {})
            stages[stage]    = status
            job.stage_status = stages
            await db.commit()
            logger.info(f"Job {job_id} — [{stage}] {status}")

    async def set_job_status(status: JobStatus, error: str = None):
        result = await db.execute(select(Job).where(Job.id == job_id))
        job    = result.scalar_one_or_none()
        if job:
            job.status = status
            if error:
                job.error_message = error
            await db.commit()

    try:
        await set_job_status(JobStatus.PROCESSING)

        await update_stage("preprocessing", "running")
        prep = run_preprocessing(file_bytes, content_type)
        await update_stage("preprocessing", "completed")

        await update_stage("segmentation", "running")
        seg = run_segmentation(prep.df_rfm)
        await update_stage("segmentation", "completed")

        # ✅ FIXED: pass both df_rfm_labelled and cluster_profiles
        await update_stage("tsne", "running")
        logger.info(f"[{job_id}] Running t-SNE...")
        tsne_result = run_tsne(seg.df_rfm_labelled, seg.cluster_profiles)
        tsne_data_serialized = json.loads(json.dumps(tsne_result.embedding_2d))
        await update_stage("tsne", "completed")

        await update_stage("association_rules", "running")
        assoc = run_association_rules(prep.df_basket)
        await update_stage("association_rules", "completed")

        await update_stage("llm_report", "running")

        analysis_data = {
            "summary":           prep.summary,
            "segments":          seg.cluster_profiles,
            "association_rules": assoc.rules,
            "trend_data":        prep.trend_data,
            "silhouette_score":  seg.silhouette_score,
        }

        prompt     = build_analysis_prompt(analysis_data, focus=focus)
        llm_result = await generate_report(prompt, model=llm_model)
        await update_stage("llm_report", "completed")

        trend_data_serialized = json.loads(json.dumps(prep.trend_data)) if prep.trend_data else None

        insight = Insight(
            id                = str(uuid.uuid4()),
            job_id            = job_id,
            summary           = prep.summary,
            cluster_profiles  = seg.cluster_profiles,
            association_rules = assoc.rules,
            n_clusters        = seg.n_clusters,
            silhouette_score  = seg.silhouette_score,
            trend_data        = trend_data_serialized,
            tsne_data         = tsne_data_serialized,  # ✅ ADDED
            llm_report        = llm_result["text"],
            model_used        = llm_result["model_used"],
        )
        db.add(insight)
        await set_job_status(JobStatus.COMPLETED)
        await db.commit()
        logger.info(f"Job {job_id} completed successfully")

    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        await set_job_status(JobStatus.FAILED, error=str(e))
        await db.commit()
        raise