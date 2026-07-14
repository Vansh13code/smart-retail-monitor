from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
import cv2
import numpy as np
import tempfile
import os
import shutil
import time

from app.services.interaction_service import InteractionService

router = APIRouter(
    prefix="/customer-analysis",
    tags=["Customer Tracking"]
)

service = InteractionService()


@router.post("/")
async def customer_analysis(
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
        task_id = create_task(video_name)
        background_tasks.add_task(process_video_background, task_id, video_path, "customer")

        return {
            "success": True,
            "message": "Video customer analysis started in the background.",
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

    result = service.process(frame)
    if isinstance(result, dict):
        result = dict(result)
        result.pop("annotated_frame", None)

    result_data = {
        "file_type": "image",
        "customer_count": result.get("total_customers", 0),
        "entry_count": len(service.entries) if service.entries else result.get("entry_count", 0),
        "exit_count": len(service.exits) if service.exits else result.get("exit_count", 0),
        "dwell_time": round(sum(c.get("dwell_time", 0.0) for c in result.get("customers", [])) / len(result.get("customers")), 1) if result.get("customers") else 0.0,
        "annotated_image": result.get("annotated_image"),
        "heatmap": result.get("heatmap"),
        "result": result
    }

    return {
        "success": True,
        "message": "Customer analysis completed successfully.",
        "data": result_data,
        "processing_time": round(time.time() - start_time, 4)
    }