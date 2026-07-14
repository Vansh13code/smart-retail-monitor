from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
import tempfile
import os

from app.services.shelf_services import ShelfService
from app.services.product_services import ProductService
import base64
import cv2
import numpy as np

router = APIRouter(
    prefix="/detect-shelf",
    tags=["Shelf Detection"]
)

shelf_service = ShelfService()
product_service = ProductService()


@router.post("/")
async def detect_shelf(file: UploadFile = File(...)):

    content_type = file.content_type or ""

    # ================= IMAGE =================
    if content_type.startswith("image/"):

        contents = await file.read()

        frame = cv2.imdecode(
            np.frombuffer(contents, np.uint8),
            cv2.IMREAD_COLOR
        )

        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image."
            )

        annotated, detections = shelf_service.process_frame(frame)

        product_analysis = product_service.process(detections)

        # encode annotated image to base64
        _, buffer = cv2.imencode('.png', annotated)
        annotated_b64 = base64.b64encode(buffer).decode('utf-8')

        return {
            "success": True,
            "file_type": "image",
            "total_detections": len(detections),
            "detections": detections,
            "annotated_image": f"data:image/png;base64,{annotated_b64}",
            "product_analysis": product_analysis,
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

            try:

                annotated, detections = shelf_service.process_frame(frame)

                product_analysis = product_service.process(detections)

                results.append({

                    "frame": frame_number,

                    "total_detections": len(detections),

                    "detections": detections,

                    "product_analysis": product_analysis

                })

            except Exception as e:

                results.append({

                    "frame": frame_number,

                    "error": str(e)

                })

        cap.release()
        os.remove(video_path)

        return {

            "success": True,

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