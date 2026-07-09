import cv2

from app.services.shelf_services import ShelfService
from app.utils.video_utils import VideoReader
from app.utils.video_utils import VideoWriter

VIDEO_PATH = "video.mp4"

reader = VideoReader(VIDEO_PATH)

service = ShelfService()

ret, frame = reader.read()

height, width = frame.shape[:2]

fps = 30

writer = VideoWriter(
    "outputs/annotated_video.mp4",
    fps,
    width,
    height
)

while ret:

    annotated_frame, shelves = service.process_frame(frame)

    print(shelves)

    writer.write(annotated_frame)

    cv2.imshow("Shelf Detection", annotated_frame)

    if cv2.waitKey(1) == ord("q"):
        break

    ret, frame = reader.read()

reader.release()

writer.release()

cv2.destroyAllWindows()