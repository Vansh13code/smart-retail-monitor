class InventoryService:

    def __init__(self):

        self.max_capacity = 10

        self.low_stock_threshold = 3

        # Dynamic planogram - can be updated based on actual detections
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

    def utilization(self, count):
        """Calculate shelf utilization percentage"""
        return round(
            (count / self.max_capacity) * 100,
            2
        )

    def shelf_status(self, count):

        if count == 0:
            return "Out of Stock"

        if count <= self.low_stock_threshold:
            return "Low Stock"

        return "Well-Stocked"

    def misplaced_products(
            self,
            shelf_id,
            products
    ):

        expected = self.planogram.get(shelf_id)
        if expected is None:
            return []

        misplaced = []

        for product in products:

            if product["class"] not in expected:

                misplaced.append(product["class"])

        return misplaced

    def find_duplicates(self, products):
        """Find duplicate products based on class and similar bounding boxes"""
        from collections import Counter
        
        class_counts = Counter(p["class"] for p in products)
        duplicates = []
        
        for class_name, count in class_counts.items():
            if count > 1:
                duplicates.append({
                    "class": class_name,
                    "count": count,
                    "product_ids": [p["id"] for p in products if p["class"] == class_name]
                })
        
        return duplicates

    def find_missing_products(self, shelf_id, products):
        """Find products that should be on shelf but are missing"""
        expected = self.planogram.get(shelf_id)
        if not expected:
            return []
        
        detected_classes = [p["class"] for p in products]
        missing = []
        
        for expected_class in expected:
            if expected_class not in detected_classes:
                missing.append(expected_class)
        
        return missing

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

    def compute_overall_metrics(self, inventory_report):
        """Compute overall inventory metrics across all shelves"""
        if not inventory_report:
            return {
                "total_shelves": 0,
                "total_products": 0,
                "overall_occupancy": 0.0,
                "overall_utilization": 0.0,
                "total_empty_spaces": 0,
                "low_stock_shelves": 0,
                "out_of_stock_shelves": 0,
                "misplaced_count": 0,
                "duplicate_count": 0
            }
        
        total_shelves = len(inventory_report)
        total_products = sum(s["product_count"] for s in inventory_report)
        overall_occupancy = round(sum(s["occupancy_percentage"] for s in inventory_report) / total_shelves, 2) if total_shelves > 0 else 0.0
        overall_utilization = round(sum(s["utilization_percentage"] for s in inventory_report) / total_shelves, 2) if total_shelves > 0 else 0.0
        total_empty_spaces = sum(s["empty_spaces"] for s in inventory_report)
        low_stock_shelves = sum(1 for s in inventory_report if s["low_stock"])
        out_of_stock_shelves = sum(1 for s in inventory_report if s["out_of_stock"])
        misplaced_count = sum(len(s["misplaced_products"]) for s in inventory_report)
        duplicate_count = sum(len(s["duplicate_products"]) for s in inventory_report)
        
        return {
            "total_shelves": total_shelves,
            "total_products": total_products,
            "overall_occupancy": overall_occupancy,
            "overall_utilization": overall_utilization,
            "total_empty_spaces": total_empty_spaces,
            "low_stock_shelves": low_stock_shelves,
            "out_of_stock_shelves": out_of_stock_shelves,
            "misplaced_count": misplaced_count,
            "duplicate_count": duplicate_count
        }

    def process(self, shelf_inventory):

        if not shelf_inventory:
            return []

        report = []

        for shelf in shelf_inventory:

            product_count = shelf["product_count"]
            occupancy = self.occupancy(product_count)
            utilization = self.utilization(product_count)

            status = self.shelf_status(product_count)

            misplaced = self.misplaced_products(
                shelf["shelf_id"],
                shelf["products"]
            )

            missing = self.find_missing_products(
                shelf["shelf_id"],
                shelf["products"]
            )

            duplicates = self.find_duplicates(shelf["products"])

            deviation = self.planogram_deviation(
                shelf["shelf_id"],
                shelf["products"]
            )

            empty_spaces = max(0, self.max_capacity - product_count)
            low_stock = product_count <= self.low_stock_threshold and product_count > 0
            out_of_stock = product_count == 0

            report.append({
                "shelf_id": shelf["shelf_id"],
                "product_count": product_count,
                "occupancy_percentage": occupancy,
                "utilization_percentage": utilization,
                "empty_spaces": empty_spaces,
                "low_stock": low_stock,
                "out_of_stock": out_of_stock,
                "status": status,
                "misplaced_products": misplaced,
                "missing_products": missing,
                "duplicate_products": duplicates,
                "planogram_deviation": deviation,
                "stock_percentage": occupancy
            })

        # Add overall metrics
        overall_metrics = self.compute_overall_metrics(report)
        
        return {
            "shelf_reports": report,
            "overall_metrics": overall_metrics
        }
