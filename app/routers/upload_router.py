from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
import os
import time

router = APIRouter(
    prefix="/upload-image",
    tags=["Upload"]
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/")
async def upload_file(file: UploadFile = File(...)):
    start_time = time.time()

    filename = file.filename
    path = os.path.join(UPLOAD_DIR, filename)

    contents = await file.read()

    with open(path, "wb") as f:
        f.write(contents)

    # ================= IMAGE =================
    if file.content_type.startswith("image/"):

        image = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file."
            )

        return {
            "success": True,
            "message": "Image uploaded successfully.",
            "data": {
                "file_type": "image",
                "filename": filename,
                "path": path,
                "shape": {
                    "height": image.shape[0],
                    "width": image.shape[1],
                    "channels": image.shape[2]
                }
            },
            "processing_time": round(time.time() - start_time, 4)
        }

    # ================= VIDEO =================
    elif file.content_type.startswith("video/"):

        cap = cv2.VideoCapture(path)

        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Unable to open video."
            )

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0

        cap.release()

        return {
            "success": True,
            "message": "Video uploaded successfully.",
            "data": {
                "file_type": "video",
                "filename": filename,
                "path": path,
                "width": width,
                "height": height,
                "fps": round(fps, 2),
                "total_frames": total_frames,
                "duration_seconds": round(duration, 2)
            },
            "processing_time": round(time.time() - start_time, 4)
        }

    # ================= INVALID FILE =================
    else:
        os.remove(path)
        raise HTTPException(
            status_code=400,
            detail="Only image and video files are supported."
        )