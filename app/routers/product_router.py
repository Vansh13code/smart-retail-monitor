from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
import tempfile
import os

from app.services.shelf_services import ShelfService
from app.services.product_services import ProductService
import time
import base64
import cv2
import numpy as np

router = APIRouter(
    prefix="/detect-products",
    tags=["Product Detection"]
)

shelf_service = ShelfService()
product_service = ProductService()


@router.post("/")
async def detect_products(file: UploadFile = File(...)):

    content_type = file.content_type or ""

    # ================= IMAGE =================
    if content_type.startswith("image/"):

        image_bytes = await file.read()

        image = cv2.imdecode(
            np.frombuffer(image_bytes, np.uint8),
            cv2.IMREAD_COLOR
        )

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image."
            )

        annotated, detections = shelf_service.process_frame(image)
        start = time.time()
        result = product_service.process(detections)
        elapsed = time.time() - start

        _, buffer = cv2.imencode('.png', annotated)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "file_type": "image",
            "result": result,
            "annotated_image": f"data:image/png;base64,{annotated_b64}",
            "execution_time": elapsed,
            "warnings": [shelf_service.detector.model_warning] if getattr(shelf_service.detector, "model_warning", None) else []
        }

    # ================= VIDEO =================
    elif content_type.startswith("video/"):

        suffix = os.path.splitext(file.filename)[1]

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(await file.read())
            video_path = temp.name

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            os.remove(video_path)
            raise HTTPException(
                status_code=400,
                detail="Unable to open video."
            )

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

            _, detections = shelf_service.process_frame(frame)

            output = product_service.process(detections)

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
            "results": results,
            "warnings": [shelf_service.detector.model_warning] if getattr(shelf_service.detector, "model_warning", None) else []
        }

    # ================= INVALID FILE =================
    else:
        raise HTTPException(
            status_code=400,
            detail="Only image and video files are supported."
        )