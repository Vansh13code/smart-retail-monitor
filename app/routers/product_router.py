from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from app.services.shelf_services import ShelfService
from app.services.product_services import ProductService

router = APIRouter(
    prefix="/detect-products",
    tags=["Product Detection"]
)

shelf_service = ShelfService()
product_service = ProductService()


@router.post("/")
async def detect_products(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    _, detections = shelf_service.process_frame(image)

    result = product_service.process(detections)

    return result