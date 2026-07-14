"""
End-to-End Validation Framework for Retail AI System
Validates detection accuracy, OCR performance, and inventory analytics
"""
import cv2
import numpy as np
import os
import json
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ValidationFramework:
    def __init__(self):
        self.results = []
        self.image_results = []
        self.video_results = []
        
        # Import services
        from app.services.shelf_services import ShelfService
        from app.services.product_services import ProductService
        from app.services.inventory_services import InventoryService
        from app.services.price_tag_detection_service import PriceTagService
        from app.services.interaction_service import InteractionService
        from app.services.pipeline_services import PipelineService
        
        self.shelf_service = ShelfService()
        self.product_service = ProductService()
        self.inventory_service = InventoryService()
        self.price_service = PriceTagService()
        self.interaction_service = InteractionService()
        self.pipeline_service = PipelineService()
        
        logger.info("Validation framework initialized with all services")
    
    def load_ground_truth(self, image_path: str) -> Dict[str, Any]:
        """
        Load ground truth labels from YOLO format if available
        Returns: dict with product_count, shelf_count, ocr_text, etc.
        """
        # Try to find corresponding label file
        image_name = Path(image_path).stem
        label_dir = Path("datasets/test/labels")
        label_file = label_dir / f"{image_name}.txt"
        
        ground_truth = {
            "product_count": 0,
            "shelf_count": 0,
            "products": [],
            "shelves": [],
            "ocr_text": ""
        }
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        # YOLO format: class_id x_center y_center width height (normalized)
                        if class_id == 0:  # Assuming 0 is shelf
                            ground_truth["shelf_count"] += 1
                        else:  # Products
                            ground_truth["product_count"] += 1
                            ground_truth["products"].append({
                                "class_id": class_id,
                                "bbox": list(map(float, parts[1:5]))
                            })
        
        return ground_truth
    
    def validate_image(self, image_path: str, ground_truth: Dict[str, Any] = None) -> Dict[str, Any]:
        """Validate a single image through the complete pipeline"""
        logger.info(f"Validating image: {image_path}")
        
        # Load image
        frame = cv2.imread(image_path)
        if frame is None:
            logger.error(f"Failed to load image: {image_path}")
            return {"error": "Failed to load image", "image_path": image_path}
        
        start_time = time.time()
        
        # Run complete pipeline
        try:
            pipeline_result = self.pipeline_service.process(frame)
            processing_time = time.time() - start_time
            
            # Extract predictions
            products = pipeline_result.get("products", {}).get("products", [])
            detections = pipeline_result.get("detections", [])
            inventory = pipeline_result.get("inventory", {})
            ocr = pipeline_result.get("ocr", {})
            
            # Count predictions
            predicted_product_count = len(products)
            predicted_shelf_count = len([d for d in detections if d.get("class", "").lower() == "shelf"])
            
            # OCR prediction
            predicted_ocr = " | ".join([d.get("text", "") for d in ocr.get("detections", [])])
            
            # If no ground truth provided, use predictions as baseline for now
            if ground_truth is None:
                ground_truth = self.load_ground_truth(image_path)
            
            # Compute metrics
            result = {
                "image_path": image_path,
                "image_name": Path(image_path).name,
                "processing_time": round(processing_time, 3),
                
                # Ground truth
                "gt_product_count": ground_truth.get("product_count", predicted_product_count),
                "gt_shelf_count": ground_truth.get("shelf_count", predicted_shelf_count),
                "gt_ocr": ground_truth.get("ocr_text", ""),
                
                # Predictions
                "predicted_product_count": predicted_product_count,
                "predicted_shelf_count": predicted_shelf_count,
                "predicted_ocr": predicted_ocr,
                
                # Accuracy
                "product_count_accuracy": 1.0 if predicted_product_count == ground_truth.get("product_count", predicted_product_count) else 0.0,
                "shelf_count_accuracy": 1.0 if predicted_shelf_count == ground_truth.get("shelf_count", predicted_shelf_count) else 0.0,
                
                # Detailed results
                "products": products,
                "detections": detections,
                "inventory": inventory,
                "ocr": ocr,
                "pipeline_result": pipeline_result,
                
                # Annotated image path (will be saved separately)
                "annotated_image": pipeline_result.get("annotated_image")
            }
            
            logger.info(f"Image validation complete: {predicted_product_count} products, {predicted_shelf_count} shelves")
            return result
            
        except Exception as e:
            logger.exception(f"Error validating image {image_path}: {e}")
            return {
                "error": str(e),
                "image_path": image_path,
                "image_name": Path(image_path).name
            }
    
    def validate_video(self, video_path: str, sample_frames: int = 10) -> Dict[str, Any]:
        """Validate a video by sampling frames"""
        logger.info(f"Validating video: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return {"error": "Failed to open video", "video_path": video_path}
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        duration = total_frames / fps
        
        # Sample frames evenly throughout video
        frame_indices = [int(i * total_frames / sample_frames) for i in range(sample_frames)]
        
        frame_results = []
        total_products = 0
        total_shelves = 0
        
        start_time = time.time()
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            try:
                pipeline_result = self.pipeline_service.process(frame)
                products = pipeline_result.get("products", {}).get("products", [])
                detections = pipeline_result.get("detections", [])
                
                frame_products = len(products)
                frame_shelves = len([d for d in detections if d.get("class", "").lower() == "shelf"])
                
                total_products += frame_products
                total_shelves += frame_shelves
                
                frame_results.append({
                    "frame_idx": frame_idx,
                    "timestamp": round(frame_idx / fps, 2),
                    "products": frame_products,
                    "shelves": frame_shelves
                })
                
            except Exception as e:
                logger.warning(f"Error processing frame {frame_idx}: {e}")
        
        cap.release()
        processing_time = time.time() - start_time
        
        avg_products = total_products / len(frame_results) if frame_results else 0
        avg_shelves = total_shelves / len(frame_results) if frame_results else 0
        
        result = {
            "video_path": video_path,
            "video_name": Path(video_path).name,
            "processing_time": round(processing_time, 3),
            "total_frames": total_frames,
            "duration_seconds": round(duration, 2),
            "sampled_frames": len(frame_results),
            
            "avg_predicted_product_count": round(avg_products, 2),
            "avg_predicted_shelf_count": round(avg_shelves, 2),
            "total_products_detected": total_products,
            "total_shelves_detected": total_shelves,
            
            "frame_results": frame_results
        }
        
        logger.info(f"Video validation complete: avg {avg_products:.1f} products, {avg_shelves:.1f} shelves per frame")
        return result
    
    def compute_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compute overall metrics from validation results"""
        if not results:
            return {}
        
        # Filter out error results
        valid_results = [r for r in results if "error" not in r]
        
        if not valid_results:
            return {"error": "No valid results to compute metrics"}
        
        # Product count accuracy
        product_accuracies = [r["product_count_accuracy"] for r in valid_results]
        avg_product_accuracy = sum(product_accuracies) / len(product_accuracies)
        
        # Shelf count accuracy
        shelf_accuracies = [r["shelf_count_accuracy"] for r in valid_results]
        avg_shelf_accuracy = sum(shelf_accuracies) / len(shelf_accuracies)
        
        # Overall accuracy
        overall_accuracy = (avg_product_accuracy + avg_shelf_accuracy) / 2
        
        # Processing time stats
        processing_times = [r["processing_time"] for r in valid_results]
        avg_processing_time = sum(processing_times) / len(processing_times)
        max_processing_time = max(processing_times)
        min_processing_time = min(processing_times)
        
        return {
            "total_images_processed": len(valid_results),
            "total_errors": len(results) - len(valid_results),
            
            "product_count_accuracy": round(avg_product_accuracy * 100, 2),
            "shelf_count_accuracy": round(avg_shelf_accuracy * 100, 2),
            "overall_accuracy": round(overall_accuracy * 100, 2),
            
            "avg_processing_time": round(avg_processing_time, 3),
            "max_processing_time": round(max_processing_time, 3),
            "min_processing_time": round(min_processing_time, 3),
            
            "total_products_detected": sum(r["predicted_product_count"] for r in valid_results),
            "total_shelves_detected": sum(r["predicted_shelf_count"] for r in valid_results),
            "avg_products_per_image": round(sum(r["predicted_product_count"] for r in valid_results) / len(valid_results), 2),
            "avg_shelves_per_image": round(sum(r["predicted_shelf_count"] for r in valid_results) / len(valid_results), 2)
        }
    
    def save_annotated_image(self, result: Dict[str, Any], output_dir: str):
        """Save annotated image from base64 or result"""
        if "annotated_image" not in result or not result["annotated_image"]:
            return None
        
        try:
            import base64
            if result["annotated_image"].startswith("data:image"):
                # Extract base64 data
                base64_data = result["annotated_image"].split(",")[1]
                image_data = base64.b64decode(base64_data)
                image_array = np.frombuffer(image_data, dtype=np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                output_path = os.path.join(output_dir, f"annotated_{result['image_name']}")
                cv2.imwrite(output_path, image)
                return output_path
        except Exception as e:
            logger.warning(f"Failed to save annotated image: {e}")
        
        return None
    
    def generate_csv_report(self, results: List[Dict[str, Any]], output_path: str):
        """Generate CSV report from validation results"""
        with open(output_path, 'w', newline='') as csvfile:
            fieldnames = [
                'image_name', 'gt_product_count', 'predicted_product_count',
                'product_count_accuracy', 'gt_shelf_count', 'predicted_shelf_count',
                'shelf_count_accuracy', 'processing_time'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            
            for result in results:
                if "error" not in result:
                    writer.writerow({
                        'image_name': result['image_name'],
                        'gt_product_count': result['gt_product_count'],
                        'predicted_product_count': result['predicted_product_count'],
                        'product_count_accuracy': result['product_count_accuracy'],
                        'gt_shelf_count': result['gt_shelf_count'],
                        'predicted_shelf_count': result['predicted_shelf_count'],
                        'shelf_count_accuracy': result['shelf_count_accuracy'],
                        'processing_time': result['processing_time']
                    })
        
        logger.info(f"CSV report saved to {output_path}")
    
    def run_validation(self, image_dir: str, video_dir: str = None):
        """Run complete validation on all images and videos"""
        logger.info("Starting complete validation run")
        
        # Create output directories
        os.makedirs("validation_output", exist_ok=True)
        os.makedirs("validation_output/annotated", exist_ok=True)
        os.makedirs("validation_output/json", exist_ok=True)
        
        # Get all images
        image_extensions = ['.jpg', '.jpeg', '.png', '.webp']
        image_files = []
        for ext in image_extensions:
            image_files.extend(Path(image_dir).glob(f"*{ext}"))
        
        logger.info(f"Found {len(image_files)} images for validation")
        
        # Validate images
        for i, image_path in enumerate(image_files[:100]):  # Limit to 100 images
            logger.info(f"Processing image {i+1}/{min(100, len(image_files))}: {image_path.name}")
            result = self.validate_image(str(image_path))
            self.image_results.append(result)
            
            # Save annotated image
            if "error" not in result:
                self.save_annotated_image(result, "validation_output/annotated")
            
            # Save JSON response
            json_path = os.path.join("validation_output/json", f"{image_path.stem}.json")
            with open(json_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
        
        # Validate videos if provided
        if video_dir:
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
            video_files = []
            for ext in video_extensions:
                video_files.extend(Path(video_dir).glob(f"*{ext}"))
            
            logger.info(f"Found {len(video_files)} videos for validation")
            
            for video_path in video_files[:3]:  # Limit to 3 videos
                logger.info(f"Processing video: {video_path.name}")
                result = self.validate_video(str(video_path))
                self.video_results.append(result)
                
                # Save JSON response
                json_path = os.path.join("validation_output/json", f"video_{video_path.stem}.json")
                with open(json_path, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
        
        # Compute metrics
        metrics = self.compute_metrics(self.image_results)
        
        # Generate reports
        self.generate_csv_report(self.image_results, "validation_output/validation_report.csv")
        
        # Save summary
        summary = {
            "image_validation": {
                "total_images": len(self.image_results),
                "metrics": metrics
            },
            "video_validation": {
                "total_videos": len(self.video_results),
                "results": self.video_results
            }
        }
        
        with open("validation_output/summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info("Validation complete. Summary saved to validation_output/summary.json")
        
        return summary


if __name__ == "__main__":
    validator = ValidationFramework()
    
    # Run validation
    summary = validator.run_validation(
        image_dir="uploads",
        video_dir="uploads"
    )
    
    print("\n" + "="*50)
    print("VALIDATION SUMMARY")
    print("="*50)
    print(json.dumps(summary, indent=2, default=str))
