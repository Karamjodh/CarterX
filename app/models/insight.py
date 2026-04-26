from sqlalchemy import Column, String, DateTime, Text, JSON, ForeignKey
from sqlalchemy.sql import func
from app.db.session import Base

class Insight(Base):
    """
    Stores all ML pipeline outputs for completed job.
    One Insight row per Job row.

    Results are stored as JSON Columns - this lets us store
    the full cluster profiles, association rules etc without
    needing separate tables for each.

    """
    __tablename__ = "insight"
    # ML Results
    id = Column(String, primary_key = True)
    job_id = Column(String, ForeignKey("jobs.id"), unique = True, nullable = False) # mapped with id of job table so that each job row has its one corresponding job insight row
    summary = Column(JSON)
    cluster_profiles = Column(JSON) # JSON format as we know has nested dictionary structure so it will help in storing
    association_rules = Column(JSON) # the multiple content values of each column attribute within a single table 
    n_clusters = Column(JSON) # instead of creating different table for cluster_profiles, association_rules etc...
    silhouette_score = Column(JSON)
    # LLM Report
    llm_report = Column(Text, nullable = True)
    model_used = Column(String, nullable = True)
    created_at = Column(DateTime(timezone = True), server_default = func.now())
