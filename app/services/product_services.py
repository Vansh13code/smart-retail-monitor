from collections import Counter
import numpy as np


class ProductService:

    def __init__(self):

        # Expanded product classes to support retail inventory
        self.product_classes = [
            "Shampoo",
            "milk",
            "snacks",
            "soft_drink",
            "Bread",
            "Beverages",
            "Chocolate",
            "Juice",
            "Biscuits",
            "Rice",
            "Oil",
            "Soap",
            "Detergent",
            "Toothpaste"
        ]

        # Accept common label variations from custom and generic detectors.
        self.alias_to_canonical = {
            "shampoo": "Shampoo",
            "milk": "milk",
            "snack": "snacks",
            "snacks": "snacks",
            "chips": "snacks",
            "softdrink": "soft_drink",
            "soft drink": "soft_drink",
            "soft_drink": "soft_drink",
            "bread": "Bread",
            "beverage": "Beverages",
            "beverages": "Beverages",
            "chocolate": "Chocolate",
            "juice": "Juice",
            "biscuit": "Biscuits",
            "biscuits": "Biscuits",
            "rice": "Rice",
            "oil": "Oil",
            "soap": "Soap",
            "detergent": "Detergent",
            "toothpaste": "Toothpaste",
            "shelf": "Shelf",
            # Fallback aliases for generic detectors when custom retail weights are unavailable.
            "bottle": "soft_drink",
            "cup": "milk",
            "person": "Person"
        }

        self._normalized_product_classes = {
            self._normalize_label(name) for name in self.product_classes
        }
        
        # Tracking ID counter for product tracking
        self._next_track_id = 1

    def _normalize_label(self, label):
        return str(label or "").strip().lower().replace("-", " ").replace("_", " ")

    def _canonicalize_label(self, label):
        normalized = self._normalize_label(label)
        return self.alias_to_canonical.get(normalized, label)

    def _normalize_detection(self, detection):
        canonical = self._canonicalize_label(detection.get("class"))
        normalized = dict(detection)
        normalized["class"] = canonical
        return normalized

    def filter_products(self, detections):

        products = []

        for detection in detections:

            normalized = self._normalize_detection(detection)
            # Accept any class that's in our product classes OR has a reasonable confidence
            normalized_label = self._normalize_label(normalized.get("class"))
            if normalized_label in self._normalized_product_classes or detection.get("confidence", 0) > 0.3:
                # Add tracking ID if not present
                if "track_id" not in normalized:
                    normalized["track_id"] = self._next_track_id
                    self._next_track_id += 1
                products.append(normalized)

        return products

    def filter_shelves(self, detections):

        shelves = []

        for detection in detections:

            normalized = self._normalize_detection(detection)
            if self._normalize_label(normalized.get("class")) == "shelf":
                shelves.append(normalized)

        return shelves

    def get_total_products(self, products):

        return len(products)

    def get_class_count(self, products):

        return dict(Counter(product["class"] for product in products))

    def get_confidence_scores(self, products):

        return [

            {
                "id": product["id"],
                "class": product["class"],
                "confidence": product["confidence"],
                "track_id": product.get("track_id", product["id"])
            }

            for product in products

        ]

    def get_bounding_boxes(self, products):

        return [

            {
                "id": product["id"],
                "class": product["class"],
                "bbox": product["bbox"],
                "track_id": product.get("track_id", product["id"])
            }

            for product in products

        ]
    
    def get_confidence_histogram(self, products, bins=10):
        """Generate confidence histogram data"""
        if not products:
            return {"bins": [], "counts": []}
        
        confidences = [p["confidence"] for p in products]
        hist, bin_edges = np.histogram(confidences, bins=bins, range=(0, 1))
        
        return {
            "bins": [round(float(edge), 2) for edge in bin_edges.tolist()],
            "counts": hist.tolist(),
            "mean_confidence": round(np.mean(confidences), 3),
            "median_confidence": round(np.median(confidences), 3),
            "std_confidence": round(np.std(confidences), 3)
        }

    def point_inside(self, product_bbox, shelf_bbox):

        px1, py1, px2, py2 = product_bbox

        sx1, sy1, sx2, sy2 = shelf_bbox

        cx = (px1 + px2) // 2
        cy = (py1 + py2) // 2

        # Retained for backwards compatibility if needed.
        return sx1 <= cx <= sx2 and sy1 <= cy <= sy2

    def map_products_to_shelves(self, shelves, products):

        shelf_inventory = {s["id"]: [] for s in shelves}
        
        for product in products:
            px1, py1, px2, py2 = product["bbox"]
            cx = (px1 + px2) // 2
            
            best_shelf_id = None
            min_dy = float('inf')
            
            # First pass: try to find the closest shelf below the product (dy >= -30)
            for shelf in shelves:
                sx1, sy1, sx2, sy2 = shelf["bbox"]
                if sx1 <= cx <= sx2:
                    dy = sy1 - py2
                    if dy >= -30 and dy < min_dy:
                        min_dy = dy
                        best_shelf_id = shelf["id"]
                        
            # Second pass fallback: if no shelf is below the product, find the closest shelf in absolute distance
            if best_shelf_id is None:
                min_abs_dy = float('inf')
                for shelf in shelves:
                    sx1, sy1, sx2, sy2 = shelf["bbox"]
                    if sx1 <= cx <= sx2:
                        dy = abs(sy1 - py2)
                        if dy < min_abs_dy:
                            min_abs_dy = dy
                            best_shelf_id = shelf["id"]
                            
            if best_shelf_id is not None:
                shelf_inventory[best_shelf_id].append(product)

        result = []
        for shelf in shelves:
            s_products = shelf_inventory[shelf["id"]]
            result.append({
                "shelf_id": shelf["id"],
                "bbox": shelf["bbox"],
                "products": s_products,
                "product_count": len(s_products)
            })

        return result

    def process(self, detections):

      shelves = self.filter_shelves(detections)

      products = self.filter_products(detections)

      # If no shelf is detected, create one virtual shelf
      if len(shelves) == 0:

          shelf_inventory = [

              {
                  "shelf_id": 1,
                  "bbox": None,
                  "products": products,
                  "product_count": len(products)
              }

          ]

      else:

          shelf_inventory = self.map_products_to_shelves(
              shelves,
              products
          )

      return {

          "total_products": self.get_total_products(products),

          "products": products,

          "class_count": self.get_class_count(products),

          "confidence_scores": self.get_confidence_scores(products),

          "bounding_boxes": self.get_bounding_boxes(products),

          "shelf_inventory": shelf_inventory,
          
          "confidence_histogram": self.get_confidence_histogram(products),

          "tracking_ids": [p.get("track_id", p["id"]) for p in products]

      }