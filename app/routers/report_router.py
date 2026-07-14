from fastapi import APIRouter, UploadFile, File, Form, BackgroundTasks, HTTPException
import cv2
import numpy as np
import os
import shutil
import time
import base64
import uuid
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
)

from app.services.pipeline_services import PipelineService

router = APIRouter(
    prefix="/generate-report",
    tags=["Reports"]
)

pipeline = PipelineService()

# ───────────────────────────── helpers ──────────────────────────────

def _encode_image(frame: np.ndarray) -> str:
    """Encode a BGR frame to a base64 PNG string."""
    success, buffer = cv2.imencode(".png", frame)
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


def _build_report(result: dict) -> dict:
    """Flatten pipeline result into a structured report dict."""
    products_data      = result.get("products") or {}
    classification_data = result.get("classification") or {}
    customers_data     = result.get("customers") or {}
    ocr_data           = result.get("ocr") or {}
    inventory_list     = result.get("inventory") or []

    # Aggregate inventory metrics across all shelves
    total_shelves = len(inventory_list)
    if total_shelves:
        avg_occupancy = sum(
            s.get("occupancy_percentage", 0) for s in inventory_list
        ) / total_shelves
        empty_spaces_total = sum(s.get("empty_spaces", 0) for s in inventory_list)
        low_stock_shelves  = sum(1 for s in inventory_list if s.get("low_stock", False))
        misplaced_products = sum(s.get("misplaced_products", 0) for s in inventory_list)
        out_of_stock       = sum(1 for s in inventory_list if s.get("out_of_stock", False))
    else:
        avg_occupancy      = 0.0
        empty_spaces_total = 0
        low_stock_shelves  = 0
        misplaced_products = 0
        out_of_stock       = 0

    # OCR detections list → price tag details
    detections = ocr_data.get("detections") or []
    price_tags = [
        {
            "label": d.get("label", ""),
            "price": d.get("price", d.get("text", ""))
        }
        for d in detections
    ]

    return {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "product_count": products_data.get("total_products", 0),
        "category_count": classification_data.get("category_count", {}),
        "inventory_summary": {
            "occupancy_percentage": round(avg_occupancy, 2),
            "empty_shelf_percentage": round(
                (empty_spaces_total / max(total_shelves, 1)) * 100, 2
            ),
            "low_stock_shelves": low_stock_shelves,
            "out_of_stock_shelves": out_of_stock,
            "misplaced_products": misplaced_products,
            "shelf_details": inventory_list,
        },
        "ocr": {
            "total_price_tags": ocr_data.get("total_price_tags", 0),
            "price_tags": price_tags,
        },
        "customers": {
            "customer_count": customers_data.get("total_customers", 0),
            "entry_count": customers_data.get("entry_count", 0),
            "exit_count": customers_data.get("exit_count", 0),
        },
    }


def _generate_pdf(report: dict, processing_time: float) -> tuple[str, str]:
    """
    Create a PDF report using reportlab and save it to the reports/ folder.

    Returns (pdf_path, pdf_url) where pdf_path is relative to cwd.
    """
    os.makedirs("reports", exist_ok=True)
    report_id  = uuid.uuid4().hex[:10]
    filename   = f"report_{report_id}.pdf"
    pdf_path   = os.path.join("reports", filename)
    pdf_url    = f"/reports/{filename}"

    doc    = SimpleDocTemplate(pdf_path, pagesize=A4,
                               rightMargin=2*cm, leftMargin=2*cm,
                               topMargin=2*cm, bottomMargin=2*cm)
    styles = getSampleStyleSheet()
    story  = []

    title_style  = styles["Title"]
    h2_style     = styles["Heading2"]
    normal_style = styles["Normal"]

    # ── Title ──
    story.append(Paragraph("Smart Retail Monitor — Report", title_style))
    story.append(Spacer(1, 0.4*cm))

    # ── Meta ──
    meta_rows = [
        ["Timestamp",       report.get("timestamp", "")],
        ["Processing Time", f"{processing_time:.4f} s"],
        ["Product Count",   str(report.get("product_count", 0))],
        ["Category Count",  str(len(report.get("category_count", {})))],
    ]
    meta_table = Table(meta_rows, colWidths=[5*cm, 11*cm])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#f0f4f8")),
        ("FONTNAME",   (0, 0), (0, -1), "Helvetica-Bold"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1), [colors.white, colors.HexColor("#fafafa")]),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 0.6*cm))

    # ── Category Breakdown ──
    story.append(Paragraph("Category Breakdown", h2_style))
    cat_count = report.get("category_count", {})
    if cat_count:
        cat_rows = [["Category", "Count"]] + [
            [k, str(v)] for k, v in cat_count.items()
        ]
        cat_table = Table(cat_rows, colWidths=[8*cm, 8*cm])
        cat_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2d6a4f")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
            ("PADDING",    (0, 0), (-1, -1), 6),
        ]))
        story.append(cat_table)
    else:
        story.append(Paragraph("No category data available.", normal_style))
    story.append(Spacer(1, 0.6*cm))

    # ── Inventory Summary ──
    story.append(Paragraph("Inventory Summary", h2_style))
    inv = report.get("inventory_summary", {})
    inv_rows = [
        ["Metric", "Value"],
        ["Occupancy (%)",          str(inv.get("occupancy_percentage", 0))],
        ["Empty Shelf (%)",        str(inv.get("empty_shelf_percentage", 0))],
        ["Low Stock Shelves",      str(inv.get("low_stock_shelves", 0))],
        ["Out-of-Stock Shelves",   str(inv.get("out_of_stock_shelves", 0))],
        ["Misplaced Products",     str(inv.get("misplaced_products", 0))],
    ]
    inv_table = Table(inv_rows, colWidths=[8*cm, 8*cm])
    inv_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1d3557")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(inv_table)
    story.append(Spacer(1, 0.6*cm))

    # ── OCR Price Tags ──
    story.append(Paragraph("OCR — Price Tags", h2_style))
    ocr = report.get("ocr", {})
    story.append(Paragraph(
        f"Total price tags detected: <b>{ocr.get('total_price_tags', 0)}</b>",
        normal_style
    ))
    price_tags = ocr.get("price_tags", [])
    if price_tags:
        pt_rows = [["Label", "Price"]] + [
            [pt.get("label", ""), str(pt.get("price", ""))]
            for pt in price_tags
        ]
        pt_table = Table(pt_rows, colWidths=[8*cm, 8*cm])
        pt_table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e63946")),
            ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
            ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#fff0f0")]),
            ("PADDING",    (0, 0), (-1, -1), 6),
        ]))
        story.append(Spacer(1, 0.3*cm))
        story.append(pt_table)
    story.append(Spacer(1, 0.6*cm))

    # ── Customer Analytics ──
    story.append(Paragraph("Customer Analytics", h2_style))
    cust = report.get("customers", {})
    cust_rows = [
        ["Metric", "Value"],
        ["Total Customers", str(cust.get("customer_count", 0))],
        ["Entry Count",     str(cust.get("entry_count", 0))],
        ["Exit Count",      str(cust.get("exit_count", 0))],
    ]
    cust_table = Table(cust_rows, colWidths=[8*cm, 8*cm])
    cust_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#457b9d")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), colors.white),
        ("FONTNAME",   (0, 0), (-1, 0), "Helvetica-Bold"),
        ("GRID",       (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f0f4f8")]),
        ("PADDING",    (0, 0), (-1, -1), 6),
    ]))
    story.append(cust_table)

    doc.build(story)
    return pdf_path, pdf_url


# ───────────────────────────── endpoint ─────────────────────────────

@router.post("/")
async def generate_report(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(None),
    filename: str = Form(None),
):
    start_time = time.time()

    # ── Determine media type ──────────────────────────────────────────
    is_video = False
    if file is not None:
        is_video = (
            file.content_type.startswith("video/") if file.content_type else False
        )
    elif filename is not None:
        is_video = os.path.splitext(filename)[1].lower() in (
            ".mp4", ".avi", ".mov", ".mkv", ".webm"
        )

    # ══════════════════════ VIDEO — background task ═══════════════════
    if is_video:
        os.makedirs("uploads", exist_ok=True)

        if file is not None:
            video_path = os.path.join("uploads", file.filename)
            with open(video_path, "wb") as f:
                f.write(await file.read())
            video_name = file.filename
        else:
            src_path = os.path.join("uploads", filename)
            if not os.path.exists(src_path):
                raise HTTPException(status_code=400, detail="Video file not found.")
            temp_name = f"temp_{int(time.time())}_{filename}"
            video_path = os.path.join("uploads", temp_name)
            shutil.copyfile(src_path, video_path)
            video_name = filename

        from app.services.video_task_service import create_task, process_video_background

        task_id = create_task(video_name)
        background_tasks.add_task(
            process_video_background, task_id, video_path, "report"
        )

        return {
            "success": True,
            "message": "Video report generation started in the background.",
            "data": {
                "task_id": task_id,
                "status": "processing",
                "progress": 0,
                "is_video": True,
            },
            "processing_time": round(time.time() - start_time, 4),
        }

    # ══════════════════════ IMAGE — full pipeline ═════════════════════
    if file is not None:
        contents = await file.read()
        frame = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file.")
    elif filename is not None:
        path = os.path.join("uploads", filename)
        if not os.path.exists(path):
            raise HTTPException(
                status_code=400, detail=f"Image file '{filename}' not found."
            )
        frame = cv2.imread(path)
        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image file.")
    else:
        raise HTTPException(status_code=400, detail="No file or filename provided.")

    # Run pipeline
    result = pipeline.process(frame)
    processing_time = round(time.time() - start_time, 4)

    # Build structured report
    report = _build_report(result)
    report["processing_time"] = processing_time

    # Generate PDF
    try:
        pdf_path, pdf_url = _generate_pdf(report, processing_time)
    except Exception as exc:
        pdf_path = ""
        pdf_url  = ""
        result.setdefault("warnings", []).append(
            f"PDF generation failed: {exc}"
        )

    # Annotated image (pipeline may return an annotated frame)
    annotated_frame = result.get("annotated_frame") or frame
    annotated_image = _encode_image(annotated_frame)

    return {
        "success": True,
        "message": "Report generated successfully.",
        "data": {
            "file_type": "image",
            "processing_time": processing_time,
            "report": report,
            "pdf_path": pdf_path,
            "pdf_url": pdf_url,
            "annotated_image": annotated_image,
            "warnings": result.get("warnings") or [],
        },
    }