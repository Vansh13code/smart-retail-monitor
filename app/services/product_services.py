from collections import Counter


class ProductService:

    def __init__(self):

        self.product_classes = [
            "Shampoo",
            "milk",
            "snacks",
            "soft_drink"
        ]

    def filter_products(self, detections):

        products = []

        for detection in detections:

            if detection["class"] in self.product_classes:
                products.append(detection)

        return products

    def filter_shelves(self, detections):

        shelves = []

        for detection in detections:

            if detection["class"] == "Shelf":
                shelves.append(detection)

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
                "confidence": product["confidence"]
            }

            for product in products

        ]

    def get_bounding_boxes(self, products):

        return [

            {
                "id": product["id"],
                "class": product["class"],
                "bbox": product["bbox"]
            }

            for product in products

        ]

    def point_inside(self, product_bbox, shelf_bbox):

        px1, py1, px2, py2 = product_bbox

        sx1, sy1, sx2, sy2 = shelf_bbox

        cx = (px1 + px2) // 2
        cy = (py1 + py2) // 2

        return sx1 <= cx <= sx2 and sy1 <= cy <= sy2

    def map_products_to_shelves(self, shelves, products):

        shelf_inventory = []

        for shelf in shelves:

            shelf_products = []

            for product in products:

                if self.point_inside(
                        product["bbox"],
                        shelf["bbox"]):

                    shelf_products.append(product)

            shelf_inventory.append({

                "shelf_id": shelf["id"],

                "bbox": shelf["bbox"],

                "products": shelf_products,

                "product_count": len(shelf_products)

            })

        return shelf_inventory

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

          "shelf_inventory": shelf_inventory

      }