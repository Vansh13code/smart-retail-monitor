import cv2

from app.services.shelf_services import ShelfService
from app.services.product_services import ProductService
from app.services.classification_services import ClassificationService
from app.services.inventory_services import InventoryService
from app.services.price_tag_detection_service import PriceTagService

from app.utils.video_utils import VideoReader
from app.utils.video_utils import VideoWriter


VIDEO_PATH = "tests/price_test3.MOV"

reader = VideoReader(VIDEO_PATH)

# ==========================================================
# Initialize Services
# ==========================================================

# Engineer 1
shelf_service = ShelfService()

# Engineer 2
product_service = ProductService()

# Engineer 3
classification_service = ClassificationService()

# Engineer 4
inventory_service = InventoryService()

# Engineer 5
price_tag_service = PriceTagService()

# ==========================================================
# Video Writer
# ==========================================================

ret, frame = reader.read()

if not ret:
    raise Exception("Unable to read video.")

height, width = frame.shape[:2]
fps = 30

writer = VideoWriter(
    "outputs/annotated_video.mp4",
    fps,
    width,
    height
)

# ==========================================================
# Processing Loop
# ==========================================================

while ret:

    # ------------------------------------------------------
    # Engineer 1 : Shelf Detection
    # ------------------------------------------------------

    annotated_frame, detections = shelf_service.process_frame(frame)

    # ------------------------------------------------------
    # Engineer 2 : Product Detection
    # ------------------------------------------------------

    product_data = product_service.process(detections)

    # ------------------------------------------------------
    # Engineer 3 : Product Classification
    # ------------------------------------------------------

    classification_data = classification_service.process(product_data)

    # ------------------------------------------------------
    # Engineer 4 : Inventory Analytics
    # ------------------------------------------------------

    inventory_data = inventory_service.process(
        product_data["shelf_inventory"]
    )

    # ------------------------------------------------------
    # Engineer 5 : Price Tag Detection Pipeline
    # ------------------------------------------------------

    price_tag_data = price_tag_service.process(frame)

    # ======================================================
    # Console Output
    # ======================================================

    print("\n================ PRODUCT REPORT ================")
    print(f"Total Products      : {product_data['total_products']}")
    print(f"Class Count         : {product_data['class_count']}")
    print(f"Confidence          : {product_data['confidence_scores']}")
    print(f"Bounding Boxes      : {product_data['bounding_boxes']}")
    print(f"Classified Products : {classification_data['classified_products']}")
    print(f"Category Count      : {classification_data['category_count']}")
    print(f"Inventory Data      : {inventory_data}")
    print("================================================")

    print("\n============= PRICE TAG REPORT =================")
    print(f"Detected Price Tags : {len(price_tag_data['detections'])}")
    print("================================================\n")

    print("\n============= OCR RESULTS =============")
    for index, result in enumerate(price_tag_data["ocr_results"]):
        print(f"\nTag {index+1}")
        print(f"Text       : {result['text']}")
        print(f"Confidence : {result['confidence']:.2f}")
    print("=======================================\n")


    print("\n============= CLEANED OCR =============")
    for i, item in enumerate(price_tag_data["cleaned_results"]):
        print(f"Tag {i+1}")
        print(item["text"])
    print("=======================================\n")


    print("\n============= PARSED RESULTS =============")
    for item in price_tag_data["parsed_results"]:
        print(item)
    print("==========================================")


    # ======================================================
    # Display
    # ======================================================

    writer.write(annotated_frame)

    cv2.imshow("Smart Retail Monitor", annotated_frame)

    # Show processed price tags
    for i, crop in enumerate(price_tag_data["processed_crops"]):

        cv2.imshow(
            f"Price Tag {i}",
            crop
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
