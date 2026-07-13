import os
import sys
import time
import json
import shutil
import random
import traceback
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).parent
DATA_DIR = ROOT / 'datasets' / 'test' / 'images'
LABEL_DIR = ROOT / 'datasets' / 'test' / 'labels'
OUT = ROOT / 'debug_output'

# utils
import cv2
import numpy as np

from app.services.price_tag_detection_service import PriceTagService
from models.detection.predict import ShelfDetector
from app.services.product_services import ProductService


def clear_output():
    if OUT.exists():
        def _onerror(func, path, excinfo):
            try:
                os.chmod(path, 0o700)
                func(path)
            except Exception:
                pass
        shutil.rmtree(OUT, onerror=_onerror)
    for s in ['original','shelf','products','price_tags','ocr','inventory','annotated','logs','reports']:
        (OUT / s).mkdir(parents=True, exist_ok=True)


def load_images(n=50):
    imgs = [p for p in DATA_DIR.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png')]
    imgs = sorted(imgs)
    random.seed(0)
    return random.sample(imgs, min(n, len(imgs)))


def read_yolo_labels(img_path, img_w, img_h):
    # label file with same stem
    lab = LABEL_DIR / (img_path.stem + '.txt')
    boxes = []
    if not lab.exists():
        return boxes
    for line in lab.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        # YOLO: class x_center y_center width height (normalized)
        _, xc, yc, w, h = map(float, parts[:5])
        x1 = int((xc - w/2) * img_w)
        y1 = int((yc - h/2) * img_h)
        x2 = int((xc + w/2) * img_w)
        y2 = int((yc + h/2) * img_h)
        boxes.append((x1,y1,x2,y2))
    return boxes


def iou(a,b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0, x2-x1) * max(0, y2-y1)
    area_a = (a[2]-a[0])*(a[3]-a[1])
    area_b = (b[2]-b[0])*(b[3]-b[1])
    union = area_a + area_b - inter
    return inter/union if union>0 else 0.0


# OCR pipelines
import cv2

def preprocess_A(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    th = cv2.adaptiveThreshold(enhanced,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,9)
    return th


def preprocess_B(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # gamma
    gamma = 1.5
    inv = 1.0/gamma
    table = np.array([((i/255.0)**inv)*255 for i in np.arange(0,256)]).astype('uint8')
    g = cv2.LUT(enhanced, table)
    # sharpen
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(g, -1, kernel)
    return sharp


def preprocess_C(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bf = cv2.bilateralFilter(gray,9,75,75)
    th = cv2.adaptiveThreshold(bf,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,9)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    morph = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return morph


def preprocess_D(img):
    # deskew + CLAHE + sharpen
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    # deskew
    coords = np.column_stack(np.where(enhanced < 255))
    if coords.shape[0] > 0:
        rect = cv2.minAreaRect(coords)
        angle = rect[-1]
        if angle < -45:
            angle = -(90 + angle)
        else:
            angle = -angle
        if abs(angle) > 0.1:
            (h,w) = enhanced.shape[:2]
            M = cv2.getRotationMatrix2D((w//2,h//2), angle, 1.0)
            enhanced = cv2.warpAffine(enhanced, M, (w,h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    sharp = cv2.filter2D(enhanced, -1, kernel)
    return sharp

PREPROCESS_PIPELINES = {
    'A': preprocess_A,
    'B': preprocess_B,
    'C': preprocess_C,
    'D': preprocess_D
}


class AutoOptimizer:
    def __init__(self, images, pipelines, max_iters=10):
        self.images = images
        self.pipelines = pipelines
        self.max_iters = max_iters
        # load immutable services once
        self.price_detector = PriceTagService()
        self.product_service = ProductService()

    def evaluate_config(self, conf, iou_thr, imgsz, ocr_pipeline):
        # Run through images and compute metrics
        results = []
        for img_path in self.images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h,w = img.shape[:2]
            # detect boxes using shelf detector as product detector proxy
            sd = ShelfDetector()
            res = sd.detect(img, conf=conf, iou=iou_thr, imgsz=imgsz)
            # convert results to box list
            pred_boxes = []
            if isinstance(res, list) and len(res)>0 and hasattr(res[0],'boxes'):
                for r in res:
                    for box in r.boxes:
                        x1,y1,x2,y2 = map(int, box.xyxy[0])
                        pred_boxes.append((x1,y1,x2,y2))
            # load ground truth
            gt_boxes = read_yolo_labels(img_path, w, h)
            # match preds to gt
            tp=0
            matched_gt=set()
            for pb in pred_boxes:
                for gi,gb in enumerate(gt_boxes):
                    if gi in matched_gt: continue
                    if iou(pb, gb) >= 0.5:
                        tp+=1
                        matched_gt.add(gi)
                        break
            fp = len(pred_boxes)-tp
            fn = len(gt_boxes)-tp
            results.append({'file':img_path.name,'tp':tp,'fp':fp,'fn':fn,'pred':len(pred_boxes),'gt':len(gt_boxes)})
        # aggregate
        total_tp = sum(r['tp'] for r in results)
        total_fp = sum(r['fp'] for r in results)
        total_fn = sum(r['fn'] for r in results)
        precision = total_tp/(total_tp+total_fp) if (total_tp+total_fp)>0 else 0.0
        recall = total_tp/(total_tp+total_fn) if (total_tp+total_fn)>0 else 0.0
        f1 = 2*precision*recall/(precision+recall) if (precision+recall)>0 else 0.0
        score = f1  # primary metric
        return {'precision':precision,'recall':recall,'f1':f1,'details':results}

    def run(self):
        # grid search over limited space
        conf_list = [0.2,0.35,0.5]
        iou_list = [0.3,0.5,0.6]
        imgsz_list = [640,960]
        ocr_list = list(self.pipelines.keys())

        best = {'score': -1}
        no_improve = 0
        iter_count = 0

        # overwrite outputs
        clear_output()

        while iter_count < self.max_iters and no_improve < 3:
            iter_count += 1
            print(f'Iteration {iter_count} starting...')
            improved = False
            for conf in conf_list:
                for iou_thr in iou_list:
                    for imgsz in imgsz_list:
                        for ocr in ocr_list:
                            try:
                                print(f'Evaluating conf={conf} iou={iou_thr} imgsz={imgsz} ocr={ocr}')
                                t0 = time.time()
                                res = self.evaluate_config(conf, iou_thr, imgsz, ocr)
                                t = time.time()-t0
                                score = res['f1']
                                print(f'-> f1={score:.4f} prec={res["precision"]:.3f} rec={res["recall"]:.3f} time={t:.1f}s')
                                if score > best['score']:
                                    best = {'score':score,'conf':conf,'iou':iou_thr,'imgsz':imgsz,'ocr':ocr,'metrics':res}
                                    improved = True
                                    no_improve = 0
                                    # Save best report
                                    with open(OUT / 'reports' / 'best_config.json','w',encoding='utf-8') as f:
                                        json.dump(best,f,indent=2)
                                else:
                                    no_improve += 0
                            except Exception as e:
                                print('Eval failed:', e)
                                traceback.print_exc()
            if not improved:
                no_improve += 1
            else:
                no_improve = 0
            print(f'Iteration {iter_count} completed. Best score so far: {best["score"]:.4f}')
        # write summary
        with open(OUT / 'reports' / 'summary_report.md','w',encoding='utf-8') as f:
            f.write('# Auto-optimizer Summary\n\n')
            f.write(f'Iterations: {iter_count}\n')
            f.write('Best config:\n')
            f.write(json.dumps(best, indent=2))
        print('Auto-optimization finished. Best:', best)


if __name__ == '__main__':
    imgs = load_images(50)
    opt = AutoOptimizer(imgs, PREPROCESS_PIPELINES, max_iters=3)
    opt.run()
