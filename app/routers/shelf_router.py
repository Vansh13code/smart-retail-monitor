from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from app.services.shelf_services import ShelfService
from app.services.product_services import ProductService

router = APIRouter(
    prefix="/detect-shelf",
    tags=["Shelf Detection"]
)

shelf_service = ShelfService()
product_service = ProductService()


@router.post("/")
async def detect_shelf(file: UploadFile = File(...)):
    contents = await file.read()

    frame = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    annotated, detections = shelf_service.process_frame(frame)

    product_analysis = product_service.process(detections)

    return {
        "success": True,
        "total_detections": len(detections),
        "detections": detections,
        "product_analysis": product_analysis
    }