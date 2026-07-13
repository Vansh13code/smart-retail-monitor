from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from app.services.interaction_service import InteractionService

router = APIRouter(
    prefix="/customer-analysis",
    tags=["Customer Tracking"]
)

service = InteractionService()


@router.post("/")
async def customer_analysis(file: UploadFile = File(...)):
    contents = await file.read()

    frame = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return {
            "success": False,
            "message": "Invalid image."
        }

    return service.process(frame)