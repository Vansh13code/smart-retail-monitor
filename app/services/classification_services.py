from collections import Counter


class ClassificationService:

    def __init__(self):

        self.category_map = {

            "shampoo": "Personal Care",

            "milk": "Dairy",

            "snacks": "Food",

            "soft drink": "Beverages",

            "soft_drink": "Beverages"

        }

    def _normalize(self, label):
        return str(label or "").strip().lower().replace("-", " ").replace("_", " ")

    def classify_products(self, products):

        classified_products = []

        for product in products:

            product_name = product.get("class", "")
            category = self.category_map.get(
                self._normalize(product_name),
                "Unknown"
            )

            classified_products.append({

                "id": product["id"],

                "product_name": product_name,

                "category": category,

                "bbox": product["bbox"],

                "confidence": product["confidence"]

            })

        return classified_products

    def category_count(self, classified_products):

        categories = []

        for product in classified_products:

            categories.append(product["category"])

        return dict(Counter(categories))

    def process(self, product_data):

        classified = self.classify_products(
            product_data["products"]
        )

        return {

            "classified_products": classified,

            "category_count":
                self.category_count(classified)

        } 