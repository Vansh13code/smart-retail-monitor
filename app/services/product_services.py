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

    def get_total_products(self, products):

        return len(products)

    def get_class_count(self, products):

        classes = []

        for product in products:

            classes.append(product["class"])

        return dict(Counter(classes))

    def get_confidence_scores(self, products):

        confidence_list = []

        for product in products:

            confidence_list.append({

                "id": product["id"],

                "class": product["class"],

                "confidence": product["confidence"]

            })

        return confidence_list

    def get_bounding_boxes(self, products):

        boxes = []

        for product in products:

            boxes.append({

                "id": product["id"],

                "class": product["class"],

                "bbox": product["bbox"]

            })

        return boxes

    def process(self, detections):

        products = self.filter_products(detections)

        output = {

            "total_products": self.get_total_products(products),

            "products": products,

            "class_count": self.get_class_count(products),

            "confidence_scores": self.get_confidence_scores(products),

            "bounding_boxes": self.get_bounding_boxes(products)

        }

        return output