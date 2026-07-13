from fastapi import FastAPI
from app.routers import health_router

app = FastAPI(
    title="Smart Retail Shelf Monitoring System",
    version="1.0.0"
)

app.include_router(health_router.router)