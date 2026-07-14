from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
import logging
import os

# Configure logging for services and pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')

from app.routers import (
    health_router,
    upload_router,
    shelf_router,
    product_router,
    classification_router,
    inventory_router,
    ocr_router,
    customer_router,
    pipeline_router,
    report_router,
    videos_router,
    search_router
)

app = FastAPI(
    title="Smart Retail Shelf Monitoring System",
    version="1.0.0"
)


@app.on_event("startup")
async def startup_event():
    """Preload models at startup for production performance"""
    try:
        from app.core.model_manager import model_manager
        model_manager.preload_all_models()
        logging.info("✓ All models preloaded successfully at startup")
    except Exception as e:
        logging.error(f"✗ Failed to preload models at startup: {e}")
        # Don't fail startup - models will load on demand

# -------------------- CORS --------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Static Files --------------------
os.makedirs("uploads", exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

os.makedirs("reports", exist_ok=True)
app.mount("/reports", StaticFiles(directory="reports"), name="reports")

# -------------------- Global Exception Handler --------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.exception("Global exception handler caught: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "success": False,
            "message": f"Server Error: {str(exc)}",
            "data": None,
            "processing_time": 0.0
        }
    )

# -------------------- Routers --------------------

app.include_router(health_router.router)
app.include_router(upload_router.router)
app.include_router(shelf_router.router)
app.include_router(product_router.router)
app.include_router(classification_router.router)
app.include_router(inventory_router.router)
app.include_router(ocr_router.router)
app.include_router(customer_router.router)
app.include_router(pipeline_router.router)
app.include_router(report_router.router)
app.include_router(videos_router.router)
app.include_router(search_router.router)
