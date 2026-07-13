from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
import tempfile
import os

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

    # ================= IMAGE =================
    if file.content_type.startswith("image/"):

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

        _, detections = shelf_service.process_frame(image)

        product_data = product_service.process(detections)

        inventory_data = inventory_service.process(
            product_data["shelf_inventory"]
        )

        return {
            "file_type": "image",
            "result": inventory_data
        }

    # ================= VIDEO =================
    elif file.content_type.startswith("video/"):

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

            product_data = product_service.process(detections)

            inventory_data = inventory_service.process(
                product_data["shelf_inventory"]
            )

            results.append({
                "frame": frame_number,
                "result": inventory_data
            })

        cap.release()
        os.remove(video_path)

        return {
            "file_type": "video",
            "video_name": file.filename,
            "frames_processed": len(results),
            "results": results
        }

    # ================= INVALID FILE =================
    else:
        raise HTTPException(
            status_code=400,
            detail="Only image and video files are supported."
        )