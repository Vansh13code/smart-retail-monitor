from collections import Counter


class ClassificationService:

    def __init__(self):

        self.category_map = {

            "Shampoo": "Personal Care",

            "milk": "Dairy",

            "snacks": "Food",

            "soft_drink": "Beverages"

        }

    def classify_products(self, products):

        classified_products = []

        for product in products:

            category = self.category_map.get(
                product["class"],
                "Unknown"
            )

            classified_products.append({

                "id": product["id"],

                "product_name": product["class"],

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