from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from app.services.price_tag_detection_service import PriceTagService

router = APIRouter(
    prefix="/ocr",
    tags=["OCR"]
)

service = PriceTagService()


@router.post("/")
async def ocr(file: UploadFile = File(...)):
    contents = await file.read()

    frame = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    return service.process(frame)