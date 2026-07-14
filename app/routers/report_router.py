from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
import tempfile
import os

from app.services.pipeline_services import PipelineService

router = APIRouter(
    prefix="/generate-report",
    tags=["Reports"]
)

pipeline = PipelineService()


@router.post("/")
async def generate_report(file: UploadFile = File(...)):

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

        result = pipeline.process(frame)

        return {
            "success": True,
            "file_type": "image",
            "report": {
                "products": result["products"]["total_products"],
                "categories": result["classification"]["category_count"],
                "inventory": result["inventory"],
                "customers": result["customers"]["total_customers"],
                "price_tags": result["ocr"]["total_price_tags"]
            }
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

        frames_processed = 0
        total_products = 0
        total_customers = 0
        total_price_tags = 0

        while True:

            success, frame = cap.read()

            if not success:
                break

            frames_processed += 1

            # Process every 10th frame
            if frames_processed % 10 != 0:
                continue

            output = pipeline.process(frame)

            total_products += output["products"]["total_products"]
            total_customers += output["customers"]["total_customers"]
            total_price_tags += output["ocr"]["total_price_tags"]

        cap.release()
        os.remove(video_path)

        return {
            "success": True,
            "file_type": "video",
            "video_name": file.filename,
            "frames_processed": frames_processed // 10,
            "report": {
                "total_products_detected": total_products,
                "total_customers_detected": total_customers,
                "total_price_tags_detected": total_price_tags
            }
        }

    # ================= INVALID =================
    else:
        raise HTTPException(
            status_code=400,
            detail="Only image and video files are supported."
        )