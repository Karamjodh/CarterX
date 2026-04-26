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

logger = logging.getLogger(__name__)

async def run_pipeline(
        job_id : str,
        file_bytes : bytes,
        content_type : str,
        db : AsyncSession,
        llm_model : str = "groq",
):
    """
    Full ML Pipeline - runs after a file is uploaded.
    
    Stages :
        1. Preprocessing  → clean data, build RFM features
        2. Segmentation   → KMeans clustering
        3. Association    → FP-Growth purchase patterns
        4. LLM Report     → AI strategy recommendations
        5. Save results   → store everything in database

    Each stage updates job.stage_status so the frontend
    can show granular progress in real time.
    """
    async def update_stage(stage : str, status : str):
        result = await db.execute(select(Job).where(Job.id == job_id)) # job_id -> Lexical Scope fxn defined within a fxn have access to data object of parent but not the other way around
        job = result.scalar_one_or_none() # returns the job if found else None
        if job:
            stages = dict(job.stage_status or {}) # either return the dict. of job status or return empty dict. (If ran for first time always empty)
            stages[stage] = status # updated the status of a stage
            job.stage_status = stages # Overwrote the stage_status attribute of job tables 
            await db.commit()
            logger.info(f"Job {job_id} - [{stage}] {status}")

    async def set_job_status(status : JobStatus, error : str = None): # error parameter default value intialized to None passing nothing in case of no error
        result = await db.execute(select(Job).where(Job.id == job_id))
        job = result.scalar_one_or_none()
        if job:
            job.status = status # overwrote the status attribute of job table
            if error:
                job.error_message = error # by default error attribute is None but in case if any error arrives it will be stored under error_message 
            await db.commit()

    try:
        await set_job_status(JobStatus.PROCESSING)
        await update_stage("preprocessing", "running")
        prep = run_preprocessing(file_bytes, content_type)
        await update_stage("preprocessing", "completed")
        await update_stage("segmentation", "running")
        seg = run_segmentation(prep.df_rfm)
        await update_stage("segmentation", "completed")
        await update_stage("association_rules","running")
        assoc = run_association_rules(prep.df_basket)
        await update_stage("association_rules","completed")
        await update_stage("llm_report", "running")
        analysis_data = {
            "summary" : prep.summary,
            "segments" : seg.cluster_profiles,
            "association_rules" : assoc.rules,
            "forecasts" : {},
        }
        prompt = build_analysis_prompt(analysis_data, focus = "general")
        llm_result = await generate_report(prompt, model = llm_model)
        await update_stage("llm_report", "completed")
        insight = Insight(
            id = str(uuid.uuid4()),
            job_id = job_id,
            summary = prep.summary,
            cluster_profiles = seg.cluster_profiles,
            association_rules = assoc.rules,
            n_clusters = seg.n_clusters,
            silhouette_score = seg.silhouette_score,
            llm_report = llm_result["text"],
            model_used = llm_result["model_used"],
        )
        db.add(insight)
        await set_job_status(JobStatus.COMPLETED)
        await db.commit()
        logger.info(f"Job {job_id} completed successfully")
    
    except Exception as e:
        logger.error(f"Job {job_id} failed: {e}")
        await set_job_status(JobStatus.FAILED, error = str(e))
        await db.commit()
        raise