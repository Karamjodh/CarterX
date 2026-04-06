from sqlalchemy import Column, String, Integer, DateTime, JSON, Enum as SQLEnum
from sqlalchemy.sql import func
import enum
import uuid

from app.db.session import Base 

class JobStatus(str, enum.Enum):
    PENDING = 'pending'
    PROCESSING = 'processing'
    COMPLETED = 'completed'
    FAILED = 'failed'

class Job(Base):
    __tablename__ = 'jobs'
    id = Column(String, primary_key = True, default = lambda : str(uuid.uuid4()))
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False)
    filename = Column(String, nullable = False)
    row_count = Column(Integer, nullable = True)
    stage_status = Column(JSON, default = dict)
    error_message = Column(String, nullable = True)
    created_at = Column(DateTime(timezone = True), server_default = func.now())
    updated_at = Column(DateTime(timezone = True), onupdate = func.now())