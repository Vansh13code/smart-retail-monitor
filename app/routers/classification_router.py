from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from app.services.shelf_services import ShelfService
from app.services.product_services import ProductService
from app.services.classification_services import ClassificationService

router = APIRouter(
    prefix="/classify-products",
    tags=["Classification"]
)

shelf_service = ShelfService()
product_service = ProductService()
classification_service = ClassificationService()


@router.post("/")
async def classify_products(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    _, detections = shelf_service.process_frame(image)

    product_data = product_service.process(detections)

    result = classification_service.process(product_data)

    return result