from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import cv2
import numpy as np
import os
import time
import base64

from app.services.shelf_services import ShelfService
from app.services.product_services import ProductService

router = APIRouter(
    prefix="/search",
    tags=["Search"]
)

shelf_service = ShelfService()
product_service = ProductService()


@router.post("/")
async def search_products(
    file: UploadFile = File(None),
    filename: str = Form(None),
    query: str = Form(None)
):
    """
    Search for products by class name and highlight them while graying out others.
    Returns count, locations, bounding boxes, and confidence of matching products.
    """
    start_time = time.time()

    if not query:
        raise HTTPException(status_code=400, detail="Search query is required.")

    # Load image
    if file is not None:
        contents = await file.read()
        frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
    elif filename is not None:
        path = os.path.join("uploads", filename)
        if not os.path.exists(path):
            raise HTTPException(status_code=400, detail=f"Image file {filename} not found.")
        frame = cv2.imread(path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file on server.")
    else:
        raise HTTPException(status_code=400, detail="No file or filename provided.")

    # Process detections
    annotated, detections = shelf_service.process_frame(frame)
    product_data = product_service.process(detections)

    # Search for matching products
    query_lower = query.lower().strip()
    matching_products = []
    other_products = []

    for product in product_data["products"]:
        class_name = product.get("class", "").lower()
        if query_lower in class_name:
            matching_products.append(product)
        else:
            other_products.append(product)

    # Create search-annotated image
    search_annotated = frame.copy()
    
    # Gray out non-matching products
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)
    
    # Start with grayed image
    search_annotated = gray_frame.copy()
    
    # Highlight matching products with color
    for product in matching_products:
        x1, y1, x2, y2 = product["bbox"]
        confidence = product["confidence"]
        
        # Restore original color in the bounding box region
        search_annotated[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
        
        # Draw green bounding box
        cv2.rectangle(search_annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Label with class name and confidence
        label = f"{product['class']} ({confidence:.2f})"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(search_annotated, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 255, 0), -1)
        cv2.putText(search_annotated, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Encode search-annotated image
    _, buffer = cv2.imencode('.png', search_annotated)
    search_annotated_b64 = base64.b64encode(buffer).decode('utf-8')

    h, w = frame.shape[:2]

    # Prepare search results
    search_results = {
        "query": query,
        "match_count": len(matching_products),
        "total_products": len(product_data["products"]),
        "matching_products": [
            {
                "id": p["id"],
                "class": p["class"],
                "confidence": p["confidence"],
                "bbox": p["bbox"],
                "track_id": p.get("track_id", p["id"])
            }
            for p in matching_products
        ],
        "other_products_count": len(other_products),
        "match_percentage": round((len(matching_products) / len(product_data["products"])) * 100, 2) if product_data["products"] else 0.0
    }

    result_data = {
        "file_type": "image",
        "model_name": "best.pt",
        "image_size": {"width": w, "height": h},
        "search_results": search_results,
        "search_annotated_image": f"data:image/png;base64,{search_annotated_b64}",
        "original_annotated_image": f"data:image/png;base64,{base64.b64encode(cv2.imencode('.png', annotated)[1]).decode('utf-8')}",
        "all_detections": product_data
    }

    return {
        "success": True,
        "message": f"Found {len(matching_products)} products matching '{query}'",
        "data": result_data,
        "processing_time": round(time.time() - start_time, 4)
    }
