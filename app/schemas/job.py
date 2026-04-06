from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from app.models.job import JobStatus

class JobCreate(BaseModel):
    filename : str
    row_count : Optional[int] = None

class JobResponse(BaseModel):
    id : str
    status : JobStatus
    filename : str
    row_count : Optional[int]
    stage_status : Optional[dict]
    error_message: Optional[str]
    created_at : Optional[datetime]
    model_config = {"from_attributes" : True}