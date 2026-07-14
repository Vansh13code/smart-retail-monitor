import uuid
import threading
import time
import os
import cv2
import logging
from typing import Optional, Dict, Any

video_tasks = {}
task_lock = threading.Lock()

def create_task(video_name: str, analysis_type: str = "pipeline") -> str:
    task_id = str(uuid.uuid4())
    with task_lock:
        video_tasks[task_id] = {
            "task_id": task_id,
            "video_name": video_name,
            "analysis_type": analysis_type,
            "status": "queued",
            "progress": 0,
            "results": [],
            "error": None,
            "start_time": time.time(),
            "processing_time": 0.0,
            "total_frames": 0,
            "processed_frames": 0,
            "current_frame": 0,
            "estimated_remaining": None,
            "can_resume": False
        }
    return task_id

def update_task_progress(task_id: str, progress: int, results: list = None, status: str = "processing", error: str = None, **kwargs):
    with task_lock:
        if task_id in video_tasks:
            video_tasks[task_id]["progress"] = progress
            video_tasks[task_id]["status"] = status
            if results is not None:
                video_tasks[task_id]["results"] = results
            if error is not None:
                video_tasks[task_id]["error"] = error
            if status in ("completed", "failed", "cancelled"):
                video_tasks[task_id]["processing_time"] = round(time.time() - video_tasks[task_id]["start_time"], 2)
            
            # Update additional fields
            for key, value in kwargs.items():
                video_tasks[task_id][key] = value

def get_task_status(task_id: str) -> Optional[Dict[str, Any]]:
    with task_lock:
        return video_tasks.get(task_id)

def cancel_task(task_id: str) -> bool:
    with task_lock:
        if task_id in video_tasks:
            if video_tasks[task_id]["status"] in ("completed", "failed", "cancelled"):
                return False
            video_tasks[task_id]["status"] = "cancelled"
            video_tasks[task_id]["error"] = "Task cancelled by user."
            return True
        return False

def cleanup_old_tasks(max_age_seconds: int = 3600):
    """Clean up tasks older than max_age_seconds to prevent memory leaks"""
    with task_lock:
        current_time = time.time()
        tasks_to_remove = []
        for task_id, task in video_tasks.items():
            if task["status"] in ("completed", "failed", "cancelled"):
                age = current_time - task["start_time"]
                if age > max_age_seconds:
                    tasks_to_remove.append(task_id)
        
        for task_id in tasks_to_remove:
            del video_tasks[task_id]
            logging.info(f"Cleaned up old task: {task_id}")

def compile_video_stats(analysis_type: str, results: list) -> dict:
    stats = {}
    total_frames = len(results)
    stats["total_frames"] = total_frames
    
    if total_frames == 0:
        return stats
        
    if analysis_type == "shelf":
        counts = [r.get("total_detections", 0) for r in results]
        stats["avg_detections"] = round(sum(counts) / total_frames, 2) if total_frames else 0
        stats["max_detections"] = max(counts) if counts else 0
        
    elif analysis_type == "product":
        counts = [r.get("result", {}).get("total_products", 0) for r in results]
        stats["avg_products"] = round(sum(counts) / total_frames, 2) if total_frames else 0
        stats["max_products"] = max(counts) if counts else 0
        
    elif analysis_type == "classification":
        counts = [r.get("result", {}).get("total_products", 0) for r in results]
        stats["avg_products"] = round(sum(counts) / total_frames, 2) if total_frames else 0
        stats["max_products"] = max(counts) if counts else 0
        
        cat_counts = {}
        for r in results:
            cats = r.get("result", {}).get("category_count", {})
            for c, v in cats.items():
                cat_counts[c] = cat_counts.get(c, 0) + v
        stats["category_counts"] = cat_counts
        
    elif analysis_type == "inventory":
        occupancies = []
        empty_spaces = []
        for r in results:
            shelf_reports = r.get("result", [])
            if shelf_reports:
                avg_occ = sum(s.get("occupancy_percentage", 0.0) for s in shelf_reports) / len(shelf_reports)
                avg_emp = sum(s.get("empty_spaces", 0) for s in shelf_reports) / len(shelf_reports)
                occupancies.append(avg_occ)
                empty_spaces.append(avg_emp)
        stats["avg_occupancy_percentage"] = round(sum(occupancies) / len(occupancies), 2) if occupancies else 0.0
        stats["avg_empty_spaces"] = round(sum(empty_spaces) / len(empty_spaces), 2) if empty_spaces else 0.0
        
    elif analysis_type == "ocr":
        prices = []
        total_tags = 0
        for r in results:
            res = r.get("result", {})
            total_tags += res.get("total_price_tags", 0)
            detections = res.get("detections", [])
            for d in detections:
                if d.get("price") is not None:
                    try:
                        prices.append(float(d["price"]))
                    except (ValueError, TypeError):
                        pass
        stats["total_price_tags_detected"] = total_tags
        stats["unique_prices"] = sorted(list(set(prices)))
        stats["avg_price"] = round(sum(prices) / len(prices), 2) if prices else 0.0
        
    elif analysis_type == "customer":
        unique_customers = set()
        dwell_times = []
        total_entries = 0
        total_exits = 0
        for r in results:
            res = r.get("result", {})
            total_entries = max(total_entries, res.get("entry_count", 0))
            total_exits = max(total_exits, res.get("exit_count", 0))
            customers = res.get("customers", [])
            for c in customers:
                unique_customers.add(c["id"])
                dwell_times.append(c.get("dwell_time", 0.0))
        stats["total_unique_customers"] = len(unique_customers)
        stats["avg_dwell_time"] = round(sum(dwell_times) / len(dwell_times), 1) if dwell_times else 0.0
        stats["max_dwell_time"] = max(dwell_times) if dwell_times else 0.0
        stats["total_entries"] = total_entries
        stats["total_exits"] = total_exits
        
    elif analysis_type in ("pipeline", "report"):
        product_counts = []
        customer_ids = set()
        dwell_times = []
        total_tags = 0
        occupancies = []
        
        for r in results:
            res = r.get("result", {})
            
            p_data = res.get("products", {})
            if isinstance(p_data, dict):
                product_counts.append(p_data.get("total_products", 0))
                
            c_data = res.get("customers", {})
            if isinstance(c_data, dict):
                for c in c_data.get("customers", []):
                    customer_ids.add(c["id"])
                    dwell_times.append(c.get("dwell_time", 0.0))
                    
            ocr_data = res.get("ocr", {})
            if isinstance(ocr_data, dict):
                total_tags += ocr_data.get("total_price_tags", 0)
                
            inv_data = res.get("inventory", [])
            if isinstance(inv_data, list) and inv_data:
                avg_occ = sum(s.get("occupancy_percentage", 0.0) for s in inv_data) / len(inv_data)
                occupancies.append(avg_occ)
                
        stats["avg_products_detected"] = round(sum(product_counts) / len(product_counts), 2) if product_counts else 0
        stats["total_unique_customers"] = len(customer_ids)
        stats["avg_dwell_time"] = round(sum(dwell_times) / len(dwell_times), 1) if dwell_times else 0.0
        stats["total_price_tags_detected"] = total_tags
        stats["avg_occupancy_percentage"] = round(sum(occupancies) / len(occupancies), 2) if occupancies else 0.0
        
    return stats

def process_video_background(task_id: str, video_path: str, analysis_type: str):
    cap = None
    out = None
    try:
        logging.info("Starting background video task %s on %s for %s", task_id, video_path, analysis_type)
        update_task_progress(task_id, 0, status="processing")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            update_task_progress(task_id, 100, status="failed", error="Unable to open video file.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration_seconds = total_frames / fps if fps > 0 else 0
        
        # Update task with video metadata
        update_task_progress(task_id, 0, total_frames=total_frames, estimated_remaining=duration_seconds)
        
        # Adaptive frame skip based on video length
        # For long videos (>5 min), skip more frames to process faster
        if duration_seconds > 300:  # 5 minutes
            frame_skip = max(2, int(fps * 2))  # Process every 2 seconds
        elif duration_seconds > 60:  # 1 minute
            frame_skip = max(1, int(fps))  # Process every 1 second
        else:
            frame_skip = max(1, int(fps / 2))  # Process every 0.5 seconds for short videos
            
        output_fps = max(1.0, fps / frame_skip)
        
        os.makedirs(os.path.join("uploads", "processed"), exist_ok=True)
        annotated_video_filename = f"annotated_{task_id}.mp4"
        annotated_video_path = os.path.join("uploads", "processed", annotated_video_filename)
        
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(annotated_video_path, fourcc, output_fps, (width, height))
        except Exception as we:
            logging.exception("Failed to initialize VideoWriter for task %s: %s", task_id, we)
            out = None
        
        results = []
        frame_idx = 0
        processed_count = 0
        last_progress_update = time.time()
        
        # Lazy imports to optimize startup
        if analysis_type == "shelf":
            from app.services.shelf_services import ShelfService
            from app.services.product_services import ProductService
            shelf_service = ShelfService()
            product_service = ProductService()
        elif analysis_type == "product":
            from app.services.shelf_services import ShelfService
            from app.services.product_services import ProductService
            shelf_service = ShelfService()
            product_service = ProductService()
        elif analysis_type == "classification":
            from app.services.shelf_services import ShelfService
            from app.services.product_services import ProductService
            from app.services.classification_services import ClassificationService
            shelf_service = ShelfService()
            product_service = ProductService()
            classification_service = ClassificationService()
        elif analysis_type == "inventory":
            from app.services.shelf_services import ShelfService
            from app.services.product_services import ProductService
            from app.services.inventory_services import InventoryService
            shelf_service = ShelfService()
            product_service = ProductService()
            inventory_service = InventoryService()
        elif analysis_type == "ocr":
            from app.services.price_tag_detection_service import PriceTagService
            price_service = PriceTagService()
        elif analysis_type == "customer":
            from app.services.interaction_service import InteractionService
            customer_service = InteractionService()
        elif analysis_type in ("pipeline", "report"):
            from app.services.pipeline_services import PipelineService
            from app.services.shelf_services import ShelfService
            pipeline_service = PipelineService()
            shelf_service = ShelfService()

        # Process frame by frame
        while True:
            # Check for cancellation request
            task_status = get_task_status(task_id)
            if task_status and task_status.get("status") == "cancelled":
                logging.info("Task %s was cancelled by user.", task_id)
                break
                
            # Check for timeout (increased to 30 minutes for long videos)
            elapsed_time = time.time() - task_status["start_time"]
            if elapsed_time > 1800:  # 30 minutes
                logging.warning("Task %s timed out after %s seconds.", task_id, elapsed_time)
                update_task_progress(task_id, 100, status="failed", error="Video processing timed out (exceeded 30 minutes).")
                break

            success, frame = cap.read()
            if not success:
                break
            
            if frame_idx % frame_skip == 0:
                try:
                    timestamp_sec = round(frame_idx / fps, 2)
                    annotated_frame = None
                    
                    if analysis_type == "shelf":
                        annotated_frame, detections = shelf_service.process_frame(frame)
                        product_analysis = product_service.process(detections)
                        frame_res = {
                            "frame": frame_idx,
                            "timestamp": timestamp_sec,
                            "total_detections": len(detections),
                            "detections": detections,
                            "product_analysis": product_analysis
                        }
                    elif analysis_type == "product":
                        annotated_frame, detections = shelf_service.process_frame(frame)
                        product_analysis = product_service.process(detections)
                        frame_res = {
                            "frame": frame_idx,
                            "timestamp": timestamp_sec,
                            "result": product_analysis
                        }
                    elif analysis_type == "classification":
                        annotated_frame, detections = shelf_service.process_frame(frame)
                        product_data = product_service.process(detections)
                        classification_data = classification_service.process(product_data)
                        frame_res = {
                            "frame": frame_idx,
                            "timestamp": timestamp_sec,
                            "result": classification_data
                        }
                    elif analysis_type == "inventory":
                        annotated_frame, detections = shelf_service.process_frame(frame)
                        product_data = product_service.process(detections)
                        inventory_data = inventory_service.process(product_data["shelf_inventory"])
                        frame_res = {
                            "frame": frame_idx,
                            "timestamp": timestamp_sec,
                            "result": inventory_data
                        }
                    elif analysis_type == "ocr":
                        ocr_data = price_service.process(frame)
                        frame_res = {
                            "frame": frame_idx,
                            "timestamp": timestamp_sec,
                            "result": ocr_data
                        }
                        # Draw price tag boxes
                        annotated_frame = frame.copy()
                        if ocr_data and "detections" in ocr_data:
                            for d in ocr_data["detections"]:
                                x1, y1, x2, y2 = d["bbox"]
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                label = f"{d.get('text', '')}"
                                cv2.putText(annotated_frame, label, (x1, max(y1 - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                    elif analysis_type == "customer":
                        customer_data = customer_service.process(frame, timestamp=timestamp_sec)
                        frame_res = {
                            "frame": frame_idx,
                            "timestamp": timestamp_sec,
                            "result": customer_data
                        }
                        annotated_frame = customer_data.get("annotated_frame")
                        if annotated_frame is None:
                            annotated_frame = frame.copy()
                    elif analysis_type in ("pipeline", "report"):
                        pipeline_data = pipeline_service.process(frame)
                        # Construct a combined annotated frame
                        annotated_frame, detections = shelf_service.process_frame(frame)
                        
                        customer_data = pipeline_data.get("customers", {})
                        if customer_data and "customers" in customer_data:
                            for c in customer_data["customers"]:
                                x1, y1, x2, y2 = c["bbox"]
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                                cv2.putText(annotated_frame, f"Customer {c['id']}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                                
                        ocr_data = pipeline_data.get("ocr", {})
                        if ocr_data and "detections" in ocr_data:
                            for d in ocr_data["detections"]:
                                x1, y1, x2, y2 = d["bbox"]
                                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                                cv2.putText(annotated_frame, f"{d.get('text', '')}", (x1, max(y1 - 8, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                
                        frame_res = {
                            "frame": frame_idx,
                            "timestamp": timestamp_sec,
                            "result": pipeline_data
                        }
                    
                    results.append(frame_res)
                    
                    if out is not None and annotated_frame is not None:
                        if annotated_frame.shape[1] != width or annotated_frame.shape[0] != height:
                            annotated_frame = cv2.resize(annotated_frame, (width, height))
                        out.write(annotated_frame)
                        
                except Exception as fe:
                    logging.exception("Error processing frame %s", frame_idx)
                    results.append({
                        "frame": frame_idx,
                        "timestamp": round(frame_idx / fps, 2),
                        "error": str(fe)
                    })
                
                processed_count += 1
                
                # Update progress every second or every 5% to avoid too frequent updates
                current_time = time.time()
                progress_pct = int((frame_idx / total_frames) * 100) if total_frames > 0 else 50
                estimated_remaining = max(0, duration_seconds - timestamp_sec) if duration_seconds > 0 else None
                
                if current_time - last_progress_update > 1.0 or progress_pct % 5 == 0:
                    update_task_progress(
                        task_id, 
                        min(99, progress_pct), 
                        results=results,
                        processed_frames=processed_count,
                        current_frame=frame_idx,
                        estimated_remaining=estimated_remaining
                    )
                    last_progress_update = current_time
            
            frame_idx += 1
            
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
            
        # If cancelled, do not mark as completed
        task_status = get_task_status(task_id)
        if task_status and task_status.get("status") == "cancelled":
            logging.info("Background video task %s cancelled cleanup done.", task_id)
            return

        overall_stats = compile_video_stats(analysis_type, results)

        final_data = {
            "success": True,
            "file_type": "video",
            "video_name": os.path.basename(video_path),
            "annotated_video_url": f"/uploads/processed/annotated_{task_id}.mp4",
            "frames_processed": len(results),
            "total_frames": total_frames,
            "results": results,
            "statistics": overall_stats,
            "processing_time": round(time.time() - task_status["start_time"], 2)
        }

        update_task_progress(task_id, 100, status="completed", results=final_data)
        logging.info("Background video task %s completed successfully", task_id)

    except Exception as e:
        logging.exception("Background video task %s failed: %s", task_id, e)
        update_task_progress(task_id, 100, status="failed", error=str(e))
    finally:
        if cap is not None:
            try:
                cap.release()
            except Exception:
                pass
        if out is not None:
            try:
                out.release()
            except Exception:
                pass
        # Clean up temporary uploaded file if we were holding a temp copy
        if "temp_" in os.path.basename(video_path) and os.path.exists(video_path):
            try:
                os.remove(video_path)
            except Exception:
                pass
