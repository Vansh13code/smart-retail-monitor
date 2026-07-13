from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np

from app.services.pipeline_services import PipelineService

router = APIRouter(
    prefix="/complete-pipeline",
    tags=["Pipeline"]
)

pipeline = PipelineService()


@router.post("/")
async def pipeline_process(file: UploadFile = File(...)):
    contents = await file.read()

    frame = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    return pipeline.process(frame)