from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from app.services.shelf_services import ShelfService
from app.services.product_services import ProductService
from app.services.inventory_services import InventoryService

router = APIRouter(
    prefix="/inventory-analysis",
    tags=["Inventory"]
)

shelf_service = ShelfService()
product_service = ProductService()
inventory_service = InventoryService()


@router.post("/")
async def inventory_analysis(file: UploadFile = File(...)):
    image_bytes = await file.read()

    image = cv2.imdecode(
        np.frombuffer(image_bytes, np.uint8),
        cv2.IMREAD_COLOR
    )

    _, detections = shelf_service.process_frame(image)

    product_data = product_service.process(detections)

    inventory_data = inventory_service.process(
        product_data["shelf_inventory"]
    )

    return inventory_data