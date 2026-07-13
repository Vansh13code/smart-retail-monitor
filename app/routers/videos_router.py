from fastapi import APIRouter, UploadFile, File
import cv2
import os

from app.services.pipeline_services import PipelineService

router = APIRouter(
    prefix="/video-analysis",
    tags=["Video Analysis"]
)

pipeline = PipelineService()


@router.post("/")
async def analyze_video(file: UploadFile = File(...)):

    os.makedirs("uploads", exist_ok=True)

    video_path = os.path.join("uploads", file.filename)

    with open(video_path, "wb") as f:
        f.write(await file.read())

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        return {
            "success": False,
            "message": "Unable to open video."
        }

    results = []
    frame_number = 0

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame_number += 1

        # Process only every 30th frame
        if frame_number % 30 != 0:
            continue

        print(f"Processing Frame {frame_number}")

        try:
            pipeline.process(frame)

            results.append({
                "frame": frame_number,
                "status": "Processed"
            })

        except Exception as e:

            results.append({
                "frame": frame_number,
                "status": "Failed",
                "error": str(e)
            })

        # Stop after 5 processed frames
        if len(results) >= 5:
            break

    cap.release()

    return {
        "success": True,
        "video": file.filename,
        "frames_processed": len(results),
        "processed_frames": results
    }