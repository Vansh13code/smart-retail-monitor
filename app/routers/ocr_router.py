from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
import tempfile
import os

from app.services.price_tag_detection_service import PriceTagService
import time
import logging
import base64
import cv2
import numpy as np

router = APIRouter(
    prefix="/ocr",
    tags=["OCR"]
)

service = PriceTagService()


@router.post("/")
async def ocr(file: UploadFile = File(...)):

    # ---------------- IMAGE ----------------
    if file.content_type.startswith("image/"):

        contents = await file.read()

        frame = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image.")

        start = time.time()
        result = service.process(frame)
        elapsed = time.time() - start

        return {
            "file_type": "image",
            "result": result,
            "execution_time": elapsed,
            "warnings": []
        }

    # ---------------- VIDEO ----------------
    elif file.content_type.startswith("video/"):

        suffix = os.path.splitext(file.filename)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:

            temp.write(await file.read())
            video_path = temp.name

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            os.remove(video_path)
            raise HTTPException(status_code=400, detail="Unable to open video.")

        results = []
        frame_number = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame_number += 1

            # Process every 10th frame
            if frame_number % 10 != 0:
                continue

            output = service.process(frame)

            results.append({
                "frame": frame_number,
                "result": output
            })

        cap.release()
        os.remove(video_path)

        return {
            "file_type": "video",
            "video_name": file.filename,
            "frames_processed": len(results),
            "results": results
        }

    # ---------------- INVALID ----------------
    else:
        raise HTTPException(
            status_code=400,
            detail="Only image and video files are supported."
        )