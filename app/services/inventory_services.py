class InventoryService:

    def __init__(self):

        self.max_capacity = 10

        self.low_stock_threshold = 3

        self.planogram = {

            1: ["milk", "milk", "snacks"],

            2: ["soft_drink", "soft_drink"],

            3: ["Shampoo"]

        }

    def occupancy(self, count):

        return round(
            (count / self.max_capacity) * 100,
            2
        )

    def shelf_status(self, count):

        if count == 0:
            return "Empty Shelf"

        if count <= self.low_stock_threshold:
            return "Low Stock"

        return "Normal"

    def misplaced_products(
            self,
            shelf_id,
            products
    ):

        expected = self.planogram.get(
            shelf_id)
        if expected is None:
            return[]
      

        misplaced = []

        for product in products:

            if product["class"] not in expected:

                misplaced.append(product["class"])

        return misplaced

    def planogram_deviation(
            self,
            shelf_id,
            products
    ):

        expected = self.planogram.get(shelf_id)
        if expected is None:
            return False
          
        expected = sorted(expected)

        detected = sorted(
            [p["class"] for p in products]
        )

        return expected != detected

    def process(self, shelf_inventory):

        report = []

        for shelf in shelf_inventory:

            occupancy = self.occupancy(
                shelf["product_count"]
            )

            status = self.shelf_status(
                shelf["product_count"]
            )

            misplaced = self.misplaced_products(

                shelf["shelf_id"],

                shelf["products"]

            )

            deviation = self.planogram_deviation(

                shelf["shelf_id"],

                shelf["products"]

            )

            report.append({

                "shelf_id":
                    shelf["shelf_id"],

                "product_count":
                    shelf["product_count"],

                "occupancy_percentage":
                    occupancy,

                "status":
                    status,

                "misplaced_products":
                    misplaced,

                "planogram_deviation":
                    deviation

            })

        return report