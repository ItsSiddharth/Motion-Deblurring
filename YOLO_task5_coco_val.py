"""
This code is a patched version of the DETR script, now using YOLO for faster and more robust benchmarking.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import random
import numpy as np
import torch
import cv2
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from PIL import Image
from ultralytics import YOLO

def set_all_seeds(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True

set_all_seeds(42)

BLUR_IMG_DIR = "/home/nam/projects/sid/Motion-Deblurring/coco_val_data_and_annots/stratified_yolo_blur_dataset_coco/test/images"
DEBLUR_IMG_DIR = "/home/nam/projects/sid/Motion-Deblurring/coco_val_data_and_annots/richardson_lucy_deblur_coco"
LABELS_DIR = "/home/nam/projects/sid/Motion-Deblurring/coco_val_data_and_annots/yolo_format_annots" # Where your .txt files are
OUTPUT_DIR = "task_5_op_coco_val_yolo_post_ft"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# model = YOLO('yolov8m.pt') 
model = YOLO("/home/nam/projects/sid/Motion-Deblurring/runs/detect/finetuning_deblur_stratified_data/yolov8m_frozen4_backbone/weights/best.pt")

def calculate_blur_score_fft(img_path, radius=30):
    img = cv2.imread(img_path, 0)
    if img is None: return 0
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2
    dft_shift = np.fft.fftshift(np.fft.fft2(img))
    mag = np.abs(dft_shift)
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)
    return np.sum(mag * (1 - mask)) / np.sum(mag * mask)

def get_yolo_outputs(image_path, threshold=0.25):
    results = model.predict(image_path, conf=threshold, verbose=False)[0]
    if len(results.boxes) == 0:
        return torch.tensor([]), torch.tensor([])
    
    boxes = results.boxes.xyxy # [x1, y1, x2, y2] in pixels
    conf = results.boxes.conf
    cls = results.boxes.cls.int()
    
    probs = torch.zeros((len(cls), 80))
    for i, (c, p) in enumerate(zip(cls, conf)):
        probs[i, min(c, 79)] = p
    return probs, boxes

def get_gt_from_txt(img_path, labels_root, img_w, img_h):
    basename = os.path.basename(img_path)
    label_path = os.path.join(labels_root, os.path.splitext(basename)[0] + ".txt")
    
    if not os.path.exists(label_path):
        return torch.tensor([]), torch.tensor([])
    
    boxes, classes = [], []
    with open(label_path, 'r') as f:
        for line in f.readlines():
            c, cx, cy, w, h = map(float, line.split())
            x1 = (cx - w/2) * img_w
            y1 = (cy - h/2) * img_h
            x2 = (cx + w/2) * img_w
            y2 = (cy + h/2) * img_h
            boxes.append([x1, y1, x2, y2])
            classes.append(int(c))
            
    if not boxes: return torch.tensor([]), torch.tensor([])
    
    probs = torch.zeros((len(classes), 80))
    for i, c in enumerate(classes):
        probs[i, min(c, 79)] = 1.0
    return probs, torch.tensor(boxes)

def calculate_iou(box1, box2):
    x1, y1 = max(box1[0], box2[0]), max(box1[1], box2[1])
    x2, y2 = min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

print("Scoring images for blur severity...")
files = [os.path.join(BLUR_IMG_DIR, f) for f in os.listdir(BLUR_IMG_DIR) if f.endswith(('.jpg', '.png'))]
scored = [[f, calculate_blur_score_fft(f)] for f in tqdm(files)]
scored = sorted(scored, key=lambda x: x[1], reverse=True)

low_idx, mid_idx = len(scored)//3, (len(scored)*2)//3
buckets = {'Low': scored[:low_idx], 'Mid': scored[low_idx:mid_idx], 'High': scored[mid_idx:]}

def benchmark_detection(gt_data, pred_data, iou_thresh=0.5):
    gt_probs, gt_boxes = gt_data
    p_probs, p_boxes = pred_data
    if len(gt_boxes) == 0: return None
    
    tp, matched = 0, set()
    for i, g_box in enumerate(gt_boxes):
        g_cls = gt_probs[i].argmax().item()
        best_iou, match_idx = 0, -1
        for j, p_box in enumerate(p_boxes):
            if j in matched: continue
            iou = calculate_iou(g_box, p_box)
            if iou > best_iou and p_probs[j].argmax().item() == g_cls:
                best_iou, match_idx = iou, j
        if best_iou >= iou_thresh:
            tp += 1
            matched.add(match_idx)
            
    fp = len(p_boxes) - len(matched)
    fn = len(gt_boxes) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return {"precision": prec, "recall": rec, "f1": f1}

def collect_mAP_data(bucket, mode='blur'):
    all_dets, total_gt = [], 0
    for img_path, _ in tqdm(bucket, desc=f"mAP {mode}"):
        with Image.open(img_path) as im: w, h = im.size
        gt_data = get_gt_from_txt(img_path, LABELS_DIR, w, h)
        if len(gt_data[1]) == 0: continue
        
        test_path = img_path if mode == 'blur' else img_path.replace("stratified_yolo_blur_dataset_coco/test/images", "richardson_lucy_deblur_coco")
        p_probs, p_boxes = get_yolo_outputs(test_path, threshold=0.01)
        
        total_gt += len(gt_data[1])
        matched = set()
        if len(p_boxes) > 0:
            confs, indices = torch.max(p_probs, dim=1)[0], torch.argsort(torch.max(p_probs, dim=1)[0], descending=True)
            for idx in indices:
                best_iou, m_idx = 0, -1
                for i, g_box in enumerate(gt_data[1]):
                    if i in matched: continue
                    if gt_data[0][i].argmax() == p_probs[idx].argmax():
                        iou = calculate_iou(g_box, p_boxes[idx])
                        if iou > best_iou: best_iou, m_idx = iou, i
                if best_iou >= 0.5:
                    all_dets.append({'conf': confs[idx].item(), 'tp': 1})
                    matched.add(m_idx)
                else:
                    all_dets.append({'conf': confs[idx].item(), 'tp': 0})
    return all_dets, total_gt

summary = []
for label, bucket in buckets.items():
    print(f"\nProcessing {label} Severity...")
    
    f1_results = {'Blur': [], 'Restored': []}
    for img_p, _ in tqdm(bucket, desc="Sample F1"):
        with Image.open(img_p) as im: w, h = im.size
        gt = get_gt_from_txt(img_p, LABELS_DIR, w, h)
        if len(gt[1]) == 0: continue
        
        f1_results['Blur'].append(benchmark_detection(gt, get_yolo_outputs(img_p)))
        f1_results['Restored'].append(benchmark_detection(gt, get_yolo_outputs(img_p.replace("stratified_yolo_blur_dataset_coco/test/images", "richardson_lucy_deblur_coco"))))
    
    b_dets, b_gt_count = collect_mAP_data(bucket, 'blur')
    r_dets, r_gt_count = collect_mAP_data(bucket, 'deblur')
    
    def calc_map(dets, total):
        if not dets: return 0
        df = pd.DataFrame(dets).sort_values('conf', ascending=False)
        tp_cs = df['tp'].cumsum().values
        recalls = tp_cs / total
        precs = tp_cs / (np.arange(len(tp_cs)) + 1)
        return np.trapz(precs, recalls)

    b_map = calc_map(b_dets, b_gt_count)
    r_map = calc_map(r_dets, r_gt_count)
    
    summary.append({'Severity': label, 'Blur_mAP': b_map, 'Restored_mAP': r_map, 'Delta': r_map - b_map})

df = pd.DataFrame(summary)
df.to_csv(f"{OUTPUT_DIR}/Final_Benchmark_Results.csv", index=False)
print("\n" + "="*30 + "\nFINAL REPORT\n" + "="*30)
print(df.to_string())

# Plotting
plt.figure(figsize=(10,6))
sns.barplot(data=df.melt(id_vars='Severity', value_vars=['Blur_mAP', 'Restored_mAP']), x='Severity', y='value', hue='variable')
plt.title("mAP Comparison: YOLO Model on Blurred vs. Restored Images")
plt.savefig(f"{OUTPUT_DIR}/mAP_Comparison.png")