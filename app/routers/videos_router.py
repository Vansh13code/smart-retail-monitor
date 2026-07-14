from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
import os
import shutil
import time
from app.services.video_task_service import create_task, get_task_status, process_video_background

router = APIRouter(
    prefix="/video-analysis",
    tags=["Video Analysis"]
)

def standard_response(success: bool, message: str, data: any, start_time: float):
    return {
        "success": success,
        "message": message,
        "data": data,
        "processing_time": round(time.time() - start_time, 4)
    }

@router.post("/")
async def analyze_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    filename: str = Form(None)
):
    start_time = time.time()
    
    # 1. Determine local video path
    if file is not None:
        os.makedirs("uploads", exist_ok=True)
        video_path = os.path.join("uploads", file.filename)
        with open(video_path, "wb") as f:
            f.write(await file.read())
        video_name = file.filename
    elif filename is not None:
        video_path = os.path.join("uploads", filename)
        if not os.path.exists(video_path):
            raise HTTPException(status_code=400, detail=f"Video file {filename} not found.")
        # Create a temp copy so the background worker doesn't delete the shared file on completion
        temp_video_name = f"temp_{int(time.time())}_{filename}"
        temp_video_path = os.path.join("uploads", temp_video_name)
        shutil.copyfile(video_path, temp_video_path)
        video_path = temp_video_path
        video_name = filename
    else:
        raise HTTPException(status_code=400, detail="No video file or filename provided.")

    # 2. Create the task state
    task_id = create_task(video_name)
    
    # 3. Add to background tasks
    background_tasks.add_task(process_video_background, task_id, video_path, "pipeline")

    return standard_response(
        True,
        "Video processing started in background.",
        {
            "task_id": task_id,
            "status": "processing",
            "progress": 0
        },
        start_time
    )

@router.get("/status/{task_id}")
async def get_video_status(task_id: str):
    start_time = time.time()
    task = get_task_status(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Video task not found.")
    
    return standard_response(
        True,
        "Task status retrieved.",
        task,
        start_time
    )

@router.post("/cancel/{task_id}")
async def cancel_video_analysis(task_id: str):
    start_time = time.time()
    from app.services.video_task_service import cancel_task
    cancelled = cancel_task(task_id)
    if not cancelled:
        raise HTTPException(status_code=400, detail="Task not found or already completed/failed.")
    return standard_response(
        True,
        "Video task cancellation requested.",
        {"task_id": task_id, "status": "cancelled"},
        start_time
    )