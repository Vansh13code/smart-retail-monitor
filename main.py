import cv2

from app.services.interaction_service import InteractionService
from app.utils.video_utils import VideoReader
from app.utils.video_utils import VideoWriter


# ==========================================================
# Video Path
# ==========================================================

VIDEO_PATH = "video3.mp4"

reader = VideoReader(VIDEO_PATH)

# ==========================================================
# Initialize Services
# ==========================================================

# Engineer 6
interaction_service = InteractionService()

# ==========================================================
# Video Writer
# ==========================================================

ret, frame = reader.read()

if not ret:
    raise Exception("Unable to read video.")

height, width = frame.shape[:2]
fps = 30

writer = VideoWriter(
    "outputs/module6_output.mp4",
    fps,
    width,
    height
)

# ==========================================================
# Processing Loop
# ==========================================================

while ret:

    # ------------------------------------------------------
    # Engineer 6 : Person Detection & Customer Tracking
    # ------------------------------------------------------

    interaction_data = interaction_service.process(frame)

    annotated_frame = interaction_data["annotated_frame"]

    customers = interaction_data["customers"]

    # ======================================================
    # Console Output
    # ======================================================

    print("\n============= CUSTOMER REPORT =============")
    print(f"Detected Customers : {len(customers)}")

    for customer in customers:

        print(f"\nCustomer ID : {customer['id']}")
        print(f"Confidence : {customer['confidence']}")
        print(f"Bounding Box : {customer['bbox']}")

    print("===========================================\n")

    # ======================================================
    # Display
    # ======================================================

    writer.write(annotated_frame)

    cv2.imshow(
        "Module 6 - Customer Detection & Tracking",
        annotated_frame
    )

    # ======================================================
    # Exit
    # ======================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    ret, frame = reader.read()

# ==========================================================
# Cleanup
# ==========================================================

reader.release()
writer.release()

cv2.destroyAllWindows()