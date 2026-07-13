from app.services.shelf_services import ShelfService
from app.services.product_services import ProductService
from app.services.classification_services import ClassificationService
from app.services.inventory_services import InventoryService
from app.services.price_tag_detection_service import PriceTagService
from app.services.interaction_service import InteractionService


class PipelineService:

    def __init__(self):
        self.shelf_service = ShelfService()
        self.product_service = ProductService()
        self.classification_service = ClassificationService()
        self.inventory_service = InventoryService()
        self.price_service = PriceTagService()
        self.interaction_service = InteractionService()

    def process(self, frame):

        # -----------------------------
        # Module 1 : Shelf Detection
        # -----------------------------
        annotated_frame, detections = self.shelf_service.process_frame(frame)

        # -----------------------------
        # Module 2 : Product Detection
        # -----------------------------
        product_data = self.product_service.process(detections)

        # -----------------------------
        # Module 3 : Classification
        # -----------------------------
        classification_data = self.classification_service.process(product_data)

        # -----------------------------
        # Module 4 : Inventory
        # -----------------------------
        inventory_data = self.inventory_service.process(
            product_data["shelf_inventory"]
        )

        # -----------------------------
        # Module 5 : OCR
        # -----------------------------
        ocr_data = self.price_service.process(frame)

        # -----------------------------
        # Module 6 : Customer Analytics
        # -----------------------------
        customer_data = self.interaction_service.process(frame)

        # -----------------------------
        # Final Response
        # -----------------------------
        return {

            "detections": detections,

            "products": product_data,

            "classification": classification_data,

            "inventory": inventory_data,

            "ocr": ocr_data,

            "customers": customer_data

        }