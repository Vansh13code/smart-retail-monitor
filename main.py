import cv2

from app.services.shelf_services import ShelfService
from app.services.product_services import ProductService
from app.services.classification_services import ClassificationService
from app.services.inventory_services import InventoryService

from app.utils.video_utils import VideoReader
from app.utils.video_utils import VideoWriter


VIDEO_PATH = "video2.mp4"

reader = VideoReader(VIDEO_PATH)

# Engineer 1
shelf_service = ShelfService()

# Engineer 2
product_service = ProductService()

# Engineer 3
classification_service = ClassificationService()

# Engineer 4
inventory_service = InventoryService()

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

    # Engineer 1
    annotated_frame, detections = shelf_service.process_frame(frame)

    # Engineer 2
    product_data = product_service.process(detections)

    # Engineer 3
    classification_data = classification_service.process(product_data)
    
    # Engineer 4
    inventory_data = inventory_service.process(product_data["shelf_inventory"])

    print("\n================ PRODUCT REPORT ================")
    print(f"Total Products : {product_data['total_products']}")
    print(f"Class Count    : {product_data['class_count']}")
    print(f"Confidence     : {product_data['confidence_scores']}")
    print(f"Bounding Boxes : {product_data['bounding_boxes']}")
    print(f"Classified Products : {classification_data['classified_products']}")
    print(f"Category Count : {classification_data['category_count']}")
    print(f"Inventory Data : {inventory_data}")
    print("================================================\n")

    writer.write(annotated_frame)

    cv2.imshow("Smart Retail Monitor", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    ret, frame = reader.read()

reader.release()
writer.release()

cv2.destroyAllWindows()