from fastapi import APIRouter, UploadFile, File, HTTPException
import cv2
import numpy as np
import tempfile
import os

from app.services.pipeline_services import PipelineService

router = APIRouter(
    prefix="/complete-pipeline",
    tags=["Pipeline"]
)

pipeline = PipelineService()


@router.post("/")
async def pipeline_process(file: UploadFile = File(...)):

    # ================= IMAGE =================
    if file.content_type.startswith("image/"):

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
            "result": result
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

            try:

                output = pipeline.process(frame)

                results.append({
                    "frame": frame_number,
                    "status": "Processed",
                    "result": output
                })

            except Exception as e:

                results.append({
                    "frame": frame_number,
                    "status": "Failed",
                    "error": str(e)
                })

        cap.release()
        os.remove(video_path)

        return {
            "success": True,
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