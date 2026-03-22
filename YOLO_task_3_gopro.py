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
os.makedirs("task_3_op", exist_ok=True)


def calculate_blur_score_fft(input_data, radius=30):
    if isinstance(input_data, str):
        img = cv2.imread(input_data, 0)
    else:
        img = cv2.cvtColor(input_data, cv2.COLOR_RGB2GRAY) if len(input_data.shape) == 3 else input_data
    
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = np.abs(dft_shift)
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)
    
    low_energy = np.sum(magnitude_spectrum * mask)
    high_energy = np.sum(magnitude_spectrum * (1 - mask))
    return high_energy / low_energy


root_dir = "gopro_deblur/blur/images"
image_and_blur_score = []

for file in tqdm(os.listdir(root_dir), desc="Scoring Blur"):
    full_path = os.path.join(root_dir, file)
    blur_score = calculate_blur_score_fft(full_path)
    image_and_blur_score.append([full_path, round(blur_score, 3)])

sorted_blur_scores = sorted(image_and_blur_score, key=lambda x: x[1], reverse=True)
low_idx, mid_idx = len(sorted_blur_scores)//3, (len(sorted_blur_scores)*2)//3

low_blur_bucket = sorted_blur_scores[:low_idx]
mid_blur_bucket = sorted_blur_scores[low_idx:mid_idx]
high_blur_bucket = sorted_blur_scores[mid_idx:]

model = YOLO('yolov8m.pt') 

def get_yolo_outputs(image_path, threshold=0.25):
    results = model.predict(image_path, conf=threshold, verbose=False)[0]
    
    if len(results.boxes) == 0:
        return torch.tensor([]), torch.tensor([])

    boxes = results.boxes.xyxy
    
    conf = results.boxes.conf
    cls = results.boxes.cls.int()
    
    num_classes = 80
    probs = torch.zeros((len(cls), num_classes))
    for i, (c, p) in enumerate(zip(cls, conf)):
        probs[i, c] = p
        
    return probs, boxes

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0

def calculate_ap(precisions, recalls):
    mpre = np.concatenate(([0.], precisions, [0.]))
    mrec = np.concatenate(([0.], recalls, [1.]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    indices = np.where(mrec[1:] != mrec[:-1])[0]
    return np.sum((mrec[indices + 1] - mrec[indices]) * mpre[indices + 1])

def benchmark_detection(gt_data, test_data, iou_thresh=0.5):
    gt_probs, gt_boxes = gt_data
    t_probs, t_boxes = test_data
    if len(gt_boxes) == 0: return None
    
    tp, matched = 0, set()
    for i, g_box in enumerate(gt_boxes):
        g_cls = gt_probs[i].argmax().item()
        best_iou, match_idx = 0, -1
        for j, t_box in enumerate(t_boxes):
            if j in matched: continue
            iou = calculate_iou(g_box, t_box)
            if iou > best_iou and t_probs[j].argmax().item() == g_cls:
                best_iou, match_idx = iou, j
        if best_iou >= iou_thresh:
            tp += 1
            matched.add(match_idx)
            
    fp = len(t_boxes) - len(matched)
    fn = len(gt_boxes) - tp
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0
    return {"precision": prec, "recall": rec, "f1": f1}

def run_full_benchmark(bucket, label):
    stats = {'Blurred': [], 'Restored': []}
    for img_path, _ in tqdm(bucket, desc=f"Benchmarking {label}"):
        sharp_path = img_path.replace("/blur", "/sharp")
        deblur_path = img_path.replace("/blur/images", "/deblur_richardson_filter")
        
        gt_data = get_yolo_outputs(sharp_path)
        if len(gt_data[1]) == 0: continue
        
        stats['Blurred'].append(benchmark_detection(gt_data, get_yolo_outputs(img_path)))
        stats['Restored'].append(benchmark_detection(gt_data, get_yolo_outputs(deblur_path)))
        
    df_b = pd.DataFrame([s for s in stats['Blurred'] if s]).mean().to_frame().T
    df_r = pd.DataFrame([s for s in stats['Restored'] if s]).mean().to_frame().T
    df_b['condition'], df_r['condition'] = 'Blurred', 'Restored'
    df_b['severity'], df_r['severity'] = label, label
    return pd.concat([df_b, df_r])

results = [run_full_benchmark(b, l) for b, l in [(low_blur_bucket, 'Low'), (mid_blur_bucket, 'Mid'), (high_blur_bucket, 'High')]]
final_report = pd.concat(results)
print(final_report)

sns.barplot(data=final_report, x='severity', y='f1', hue='condition')
plt.savefig("task_3_op/yolo_f1_comparison.png")
plt.close()

def collect_global_detections(bucket, mode='blur'):
    all_dets, total_gt = [], 0
    for img_path, _ in tqdm(bucket, desc=f"Global {mode}"):
        sharp_path = img_path.replace("/blur", "/sharp")
        test_path = img_path if mode == 'blur' else img_path.replace("/blur/images", "/deblur_richardson_filter")
        
        gt_probs, gt_boxes = get_yolo_outputs(sharp_path, threshold=0.3)
        t_probs, t_boxes = get_yolo_outputs(test_path, threshold=0.01)
        
        total_gt += len(gt_boxes)
        matched = set()
        
        if len(t_boxes) > 0:
            confs, indices = torch.max(t_probs, dim=1)[0], torch.argsort(torch.max(t_probs, dim=1)[0], descending=True)
            for idx in indices:
                best_iou, match_idx = 0, -1
                for i, g_box in enumerate(gt_boxes):
                    if i in matched: continue
                    if gt_probs[i].argmax() == t_probs[idx].argmax():
                        iou = calculate_iou(g_box, t_boxes[idx])
                        if iou > best_iou: best_iou, match_idx = iou, i
                
                if best_iou >= 0.5:
                    all_dets.append({'conf': confs[idx].item(), 'tp': 1})
                    matched.add(match_idx)
                else:
                    all_dets.append({'conf': confs[idx].item(), 'tp': 0})
    return all_dets, total_gt

def compute_map(detections, total_gt):
    df = pd.DataFrame(detections).sort_values('conf', ascending=False)
    tp_cs, fp_cs = df['tp'].cumsum().values, (1 - df['tp']).cumsum().values
    recalls = tp_cs / total_gt
    precisions = tp_cs / (tp_cs + fp_cs)
    return calculate_ap(precisions, recalls), precisions, recalls


summary = []
for bucket, label in [(low_blur_bucket, 'Low'), (mid_blur_bucket, 'Mid'), (high_blur_bucket, 'High')]:
    b_ap, _, _ = compute_map(*collect_global_detections(bucket, 'blur'))
    r_ap, _, _ = compute_map(*collect_global_detections(bucket, 'deblur'))
    summary.append({'Severity': label, 'Blur mAP': b_ap, 'Restored mAP': r_ap, 'Delta': r_ap - b_ap})

pd.DataFrame(summary).to_csv("task_3_op/YOLO_inference_task3.csv")
print("Benchmarking Complete. Files saved to task_3_op/")