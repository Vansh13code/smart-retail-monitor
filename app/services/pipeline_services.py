import time
import logging

from app.services.shelf_services import ShelfService
from app.services.product_services import ProductService
from app.services.classification_services import ClassificationService
from app.services.inventory_services import InventoryService
from app.services.price_tag_detection_service import PriceTagService
from app.services.interaction_service import InteractionService


class PipelineService:

    def __init__(self):
        # Initialize services once (models loaded in their constructors)
        self.shelf_service = ShelfService()
        self.product_service = ProductService()
        self.classification_service = ClassificationService()
        self.inventory_service = InventoryService()
        self.price_service = PriceTagService()
        self.interaction_service = InteractionService()

    def process(self, frame):
        response = {
            "detections": None,
            "products": None,
            "classification": None,
            "inventory": None,
            "ocr": None,
            "customers": None,
            "warnings": [],
            "timings": {},
        }

        # Stage helper
        def run_stage(name, fn, *args, **kwargs):
            start = time.time()
            try:
                out = fn(*args, **kwargs)
                elapsed = time.time() - start
                response["timings"][name] = elapsed
                return out
            except Exception as e:
                elapsed = time.time() - start
                logging.exception("Stage %s failed: %s", name, e)
                response["warnings"].append({"stage": name, "error": str(e)})
                response["timings"][name] = elapsed
                return None

        # -----------------------------
        # Module 1 : Shelf Detection
        # -----------------------------
        shelf_out = run_stage("shelf_detection", self.shelf_service.process_frame, frame)
        if shelf_out:
            annotated_frame, detections = shelf_out
            response["detections"] = detections
        else:
            annotated_frame, detections = frame, []

        # -----------------------------
        # Module 2 : Product Detection
        # -----------------------------
        product_data = run_stage("product_detection", self.product_service.process, detections) or {}
        response["products"] = product_data

        # -----------------------------
        # Module 3 : Classification
        # -----------------------------
        classification_data = run_stage("classification", self.classification_service.process, product_data) or {}
        response["classification"] = classification_data

        # -----------------------------
        # Module 4 : Inventory
        # -----------------------------
        shelf_inventory = product_data.get("shelf_inventory") if isinstance(product_data, dict) else None
        inventory_data = run_stage("inventory", self.inventory_service.process, shelf_inventory) or {}
        response["inventory"] = inventory_data

        # -----------------------------
        # Module 5 : OCR
        # -----------------------------
        ocr_data = run_stage("ocr", self.price_service.process, frame) or {}
        response["ocr"] = ocr_data

        # -----------------------------
        # Module 6 : Customer Analytics
        # -----------------------------
        customer_data = run_stage("customers", self.interaction_service.process, frame) or {}
        response["customers"] = customer_data

        # Final response
        return response