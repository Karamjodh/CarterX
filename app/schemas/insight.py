from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class InsightResponse(BaseModel):
    """
    What the API returns when someone fetches insights for a job.
    """
    model_config = {"from_attributes" : True, "protected_namespaces" : ()}
    id : str
    job_id : str
    summary : Optional[dict] = None
    cluster_profiles : Optional[list] = None
    association_rules : Optional[list] = None
    n_clusters : Optional[int] = None
    silhouette_score : Optional[float] = None
    llm_report : Optional[str] = None
    model_used : Optional[str] = None
    created_at : Optional[datetime] = None
