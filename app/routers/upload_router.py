from fastapi import APIRouter, UploadFile, File
import cv2
import numpy as np
import uuid
import os

router = APIRouter(
    prefix="/upload-image",
    tags=["Upload"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/")
async def upload_image(file: UploadFile = File(...)):
    contents = await file.read()

    filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(UPLOAD_DIR, filename)

    with open(path, "wb") as f:
        f.write(contents)

    image = cv2.imdecode(
        np.frombuffer(contents, np.uint8),
        cv2.IMREAD_COLOR
    )

    return {
        "success": True,
        "filename": filename,
        "path": path,
        "shape": image.shape
    }