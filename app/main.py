from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.core.config import settings
from app.db.session import init_db
from app.api.routes import health, jobs, uploads,reports,insights

@asynccontextmanager
async def lifespan(app : FastAPI):
    await init_db()
    print("Database Ready")
    yield
    print("Shutting down...")
    
app = FastAPI(
    title = settings.APP_NAME,
    description = "AI-powered customer analytics platform",
    version = "1.0.0",
    lifespan = lifespan,
)
app.include_router(health.router, prefix = "/api/v1", tags = ["Health"])
app.include_router(jobs.router, prefix = "/api/v1/jobs", tags = ["Jobs"])
app.include_router(uploads.router, prefix = "/api/v1/uploads", tags = ["Uploads"])
app.include_router(reports.router, prefix = "/api/v1/reports", tags = ["Reports"])
app.include_router(insights.router, prefix = "/api/v1/insights", tags = ["Insights"])
@app.get("/")
def root():
    return {
        'app' : settings.APP_NAME,
        'environment' : settings.ENVIORNMENT,
        'message' : "Welcome to Carter.ai",
        'docs' : '/docs'
    }