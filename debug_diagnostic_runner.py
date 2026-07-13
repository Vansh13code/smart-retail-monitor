import os
import sys
import random
import json
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'datasets' / 'test' / 'images'
OUT = ROOT / 'debug_output'

# Create output dirs
structure = ['original','shelf','products','price_tags','ocr','inventory','annotated','logs','reports']
for s in structure:
    (OUT / s).mkdir(parents=True, exist_ok=True)

# pick images
images = [p for p in DATA_DIR.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')]
if not images:
    print('No images found in', DATA_DIR)
    sys.exit(1)

random.seed(0)
selected = random.sample(images, min(50, len(images)))

# Try to import services
services = {}
imports = {
    'price': 'app.services.price_tag_detection_service.PriceTagService',
    'shelf': 'app.services.shelf_services.ShelfService',
    'product': 'app.services.product_services.ProductService',
    'pipeline': 'app.services.pipeline_services.PipelineService',
}

for k, path in imports.items():
    try:
        modpath, cls = path.rsplit('.', 1)
        mod = __import__(modpath, fromlist=[cls])
        services[k] = getattr(mod, cls)()
        print('Loaded', k, 'service')
    except Exception as e:
        services[k] = None
        print('Could not load', k, 'service:', e)

report_rows = []

for idx, img_path in enumerate(selected, 1):
    start_t = time.time()
    try:
        img = None
        import cv2
        img = cv2.imread(str(img_path))
        if img is None:
            raise ValueError('cv2 failed to load image')

        # save original copy
        orig_out = OUT / 'original' / img_path.name
        cv2.imwrite(str(orig_out), img)

        per_image = {
            'file': img_path.name,
            'shelf': None,
            'products': None,
            'price_tags': None,
            'ocr': None,
            'inventory': None,
            'errors': []
        }

        # shelf
        if services.get('shelf'):
            try:
                res = services['shelf'].process_frame(img)
                # process_frame assumed to return (annotated, detections)
                annotated, detections = res if isinstance(res, tuple) and len(res) == 2 else (None, res)
                per_image['shelf'] = {'detections': detections}
                if annotated is not None:
                    outp = OUT / 'shelf' / img_path.name
                    cv2.imwrite(str(outp), annotated)
            except Exception as e:
                per_image['errors'].append('shelf:'+str(e))

        # product: use detections from shelf stage if available
        if services.get('product'):
            try:
                shelf_dets = per_image.get('shelf', {}).get('detections') or []
                # ProductService.process expects detections list
                res = services['product'].process(shelf_dets)
                per_image['products'] = res
                # if returns annotated image (not expected), save
                if isinstance(res, dict) and 'annotated_image' in res and res['annotated_image']:
                    import base64
                    header, b64 = res['annotated_image'].split(',',1) if ',' in res['annotated_image'] else (None, res['annotated_image'])
                    data = base64.b64decode(b64)
                    with open(OUT / 'products' / img_path.name, 'wb') as f:
                        f.write(data)
            except Exception as e:
                per_image['errors'].append('product:'+str(e))

        # price tags
        if services.get('price'):
            try:
                res = services['price'].process(img)
                per_image['price_tags'] = res
                if res and res.get('annotated_image'):
                    import base64
                    header, b64 = res['annotated_image'].split(',',1) if ',' in res['annotated_image'] else (None, res['annotated_image'])
                    data = base64.b64decode(b64)
                    with open(OUT / 'price_tags' / img_path.name, 'wb') as f:
                        f.write(data)
            except Exception as e:
                per_image['errors'].append('price:'+str(e))

        # pipeline (classification + inventory)
        if services.get('pipeline'):
            try:
                res = services['pipeline'].process(img)
                per_image['inventory'] = res
            except Exception as e:
                per_image['errors'].append('pipeline:'+str(e))

        # OCR raw saving placeholder: if price tag service provided OCR blocks, save
        if per_image['price_tags'] and isinstance(per_image['price_tags'], dict):
            try:
                ocrs = per_image['price_tags'].get('ocr_results')
                with open(OUT / 'ocr' / (img_path.stem + '.json'), 'w', encoding='utf-8') as f:
                    json.dump(ocrs, f, indent=2, default=str)
                # cleaned
                cleaned = per_image['price_tags'].get('cleaned_results')
                with open(OUT / 'ocr' / (img_path.stem + '_cleaned.json'), 'w', encoding='utf-8') as f:
                    json.dump(cleaned, f, indent=2, default=str)
            except Exception as e:
                per_image['errors'].append('ocrsave:'+str(e))

        # final annotated: prefer pipeline annotated, then price_tags annotated, then products
        annotated_written = False
        for key in ('inventory','price_tags','products'):
            v = per_image.get(key)
            if isinstance(v, dict) and v.get('annotated_image'):
                try:
                    import base64
                    header, b64 = v['annotated_image'].split(',',1) if ',' in v['annotated_image'] else (None, v['annotated_image'])
                    data = base64.b64decode(b64)
                    with open(OUT / 'annotated' / img_path.name, 'wb') as f:
                        f.write(data)
                    annotated_written = True
                    break
                except Exception as e:
                    per_image['errors'].append('annotated_save:'+str(e))
        # if none, save shelf annotated if exists
        if not annotated_written and services.get('shelf'):
            try:
                res = services['shelf'].process_frame(img)
                annotated, _ = res if isinstance(res, tuple) and len(res) == 2 else (None, None)
                if annotated is not None:
                    cv2.imwrite(str(OUT / 'annotated' / img_path.name), annotated)
            except Exception:
                pass

        # write per-image json
        with open(OUT / 'logs' / (img_path.stem + '.json'), 'w', encoding='utf-8') as f:
            json.dump(per_image, f, indent=2, default=str)

        # summary row
        elapsed = time.time() - start_t
        report_rows.append({
            'file': img_path.name,
            'time_s': round(elapsed,2),
            'errors': per_image['errors'],
            'shelf_count': len(per_image['shelf']['detections']) if per_image.get('shelf') and per_image['shelf'].get('detections') else 0,
            'product_count': len(per_image['products']['products']) if per_image.get('products') and isinstance(per_image['products'], dict) and per_image['products'].get('products') else 0,
        })

    except Exception as e:
        print('Image', img_path.name, 'failed:', e)
        traceback.print_exc()
        with open(OUT / 'logs' / (img_path.stem + '_error.txt'), 'w', encoding='utf-8') as f:
            f.write(traceback.format_exc())

# write summary csv and md
import csv
with open(OUT / 'reports' / 'debug_report.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.DictWriter(f, fieldnames=['file','time_s','shelf_count','product_count','errors'])
    writer.writeheader()
    for r in report_rows:
        writer.writerow({
            'file': r['file'], 'time_s': r['time_s'], 'shelf_count': r.get('shelf_count',0), 'product_count': r.get('product_count',0), 'errors': ';'.join(r['errors'])
        })

with open(OUT / 'reports' / 'summary_report.md', 'w', encoding='utf-8') as f:
    f.write('# Diagnostic Summary\n\n')
    f.write(f'Processed {len(report_rows)} images.\n\n')
    f.write('## Failures\n')
    for r in report_rows:
        if r['errors']:
            f.write(f"- {r['file']}: {r['errors']}\n")

print('Done. Outputs saved to', OUT)
