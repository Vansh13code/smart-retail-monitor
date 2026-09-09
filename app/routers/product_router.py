from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
import cv2
import numpy as np
import tempfile
import os
import shutil
import time
import base64

from app.services.shelf_services import ShelfService
from app.services.product_services import ProductService

router = APIRouter(
    prefix="/detect-products",
    tags=["Product Detection"]
)

shelf_service = ShelfService()
product_service = ProductService()


@router.post("/")
async def detect_products(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    filename: str = Form(None)
):
    start_time = time.time()

    # Determine if request is video
    is_video = False
    if file is not None:
        is_video = file.content_type.startswith("video/") if file.content_type else False
    elif filename is not None:
        is_video = os.path.splitext(filename)[1].lower() in (".mp4", ".avi", ".mov", ".mkv", ".mov")

    # ================= VIDEO BACKGROUND PROCESSING =================
    if is_video:
        if file is not None:
            os.makedirs("uploads", exist_ok=True)
            video_path = os.path.join("uploads", file.filename)
            with open(video_path, "wb") as f:
                f.write(await file.read())
            video_name = file.filename
        else:
            video_path = os.path.join("uploads", filename)
            if not os.path.exists(video_path):
                raise HTTPException(status_code=400, detail="Video file not found.")
            # copy temp
            temp_name = f"temp_{int(time.time())}_{filename}"
            temp_path = os.path.join("uploads", temp_name)
            shutil.copyfile(video_path, temp_path)
            video_path = temp_path
            video_name = filename


        from app.services.video_task_service import create_task, process_video_background
        task_id = create_task(video_name, "product")
        background_tasks.add_task(process_video_background, task_id, video_path, "product")

        return {
            "success": True,
            "message": "Video product detection started in the background.",
            "data": {
                "task_id": task_id,
                "status": "processing",
                "progress": 0,
                "is_video": True
            },
            "processing_time": round(time.time() - start_time, 4)
        }

    # ================= IMAGE PROCESSING =================
    if file is not None:
        contents = await file.read()
        frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
    elif filename is not None:
        path = os.path.join("uploads", filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=400, detail=f"Image file {filename} not found.")
        frame = cv2.imread(path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file on server.")
    else:
        raise HTTPException(status_code=400, detail="No file or filename provided.")

    annotated, detections = shelf_service.process_frame(frame)
    result = product_service.process(frame,detections)

    # Product overlays are blue; ShelfService has already drawn shelves green.
    for product in result["products"]:
        x1, y1, x2, y2 = product["bbox"]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{product['class']} ({product['confidence']:.2f})"
        cv2.putText(annotated, label, (x1, max(y1 - 7, 18)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 0), 2)

    _, buffer = cv2.imencode('.png', annotated)
    annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    h, w = frame.shape[:2]

    result_data = {
        "file_type": "image",
        "model_name": "best.pt",
        "image_size": {"width": w, "height": h},
        "product_count": result["total_products"],
        "confidence": round(sum(p["confidence"] for p in result["products"]) / len(result["products"]), 2) if result["products"] else 0.0,
        "classes": list(result["class_count"].keys()),
        "class_count": result["class_count"],
        "confidence_scores": result["confidence_scores"],
        "bounding_boxes": result["bounding_boxes"],
        "tracking_ids": result.get("tracking_ids", []),
        "confidence_histogram": result.get("confidence_histogram", {}),
        "shelf_inventory": result["shelf_inventory"],
        "annotated_image": f"data:image/png;base64,{annotated_b64}",
        "result": result,
        "warnings": [shelf_service.detector.model_warning] if getattr(shelf_service.detector, "model_warning", None) else []
    }

    return {
        "success": True,
        "message": "Product detection completed successfully.",
        "data": result_data,
        "processing_time": round(time.time() - start_time, 4)
    }
