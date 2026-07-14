import time
import logging
import cv2
import base64
import os

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
            "annotated_image": None,
            "model_name": "best.pt",
            "image_size": None
        }

        # Store image dimensions
        h, w = frame.shape[:2]
        response["image_size"] = {"width": w, "height": h}

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

        # -----------------------------
        # Enhanced Annotation with all information
        # -----------------------------
        try:
            enhanced_annotated = self.shelf_service.annotate_with_inventory(
                frame, 
                detections, 
                shelf_inventory or [], 
                ocr_data
            )
            
            # Encode annotated image
            _, buffer = cv2.imencode('.png', enhanced_annotated)
            annotated_b64 = base64.b64encode(buffer).decode('utf-8')
            response["annotated_image"] = f"data:image/png;base64,{annotated_b64}"
            
            # Save annotated image to outputs/
            os.makedirs("outputs", exist_ok=True)
            output_path = os.path.join("outputs", f"pipeline_output_{int(time.time())}.png")
            cv2.imwrite(output_path, enhanced_annotated)
            response["annotated_image_path"] = output_path
            
        except Exception as e:
            logging.exception("Enhanced annotation failed: %s", e)
            response["warnings"].append({"stage": "annotation", "error": str(e)})
            # Fallback to basic annotation
            if annotated_frame is not None:
                _, buffer = cv2.imencode('.png', annotated_frame)
                annotated_b64 = base64.b64encode(buffer).decode('utf-8')
                response["annotated_image"] = f"data:image/png;base64,{annotated_b64}"

        # Final response
        return response