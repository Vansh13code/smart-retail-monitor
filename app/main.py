from fastapi import FastAPI

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
    videos_router
)

app = FastAPI(
    title="Smart Retail Shelf Monitoring System",
    version="1.0.0"
)

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