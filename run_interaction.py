import cv2

from app.services.interaction_service import InteractionService

VIDEO_PATH = "video2.mp4"      # Change if needed

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise Exception("Cannot open video.")

interaction_service = InteractionService()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    # Module 6
    interaction_data = interaction_service.process(frame)

    annotated_frame = interaction_data["annotated_frame"]
    customers = interaction_data["customers"]

    print("\n============= CUSTOMER REPORT =============")
    print(f"Detected Customers : {len(customers)}")

    for customer in customers:
        print(
            f"Customer ID : {customer['id']} | "
            f"Confidence : {customer['confidence']} | "
            f"BBox : {customer['bbox']}"
        )

    cv2.imshow(
        "Module 6 - Customer Detection",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()