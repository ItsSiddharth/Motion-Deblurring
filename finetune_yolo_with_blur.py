# Code block generated with Gemini; prompt "All seeds needed for reproducibility in pytorch based DL project"

import os
import random
import numpy as np
import torch

def set_all_seeds(seed_value=42):
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    random.seed(seed_value)
    np.random.seed(seed_value)
    
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

set_all_seeds(42)

from ultralytics import YOLO

model = YOLO('yolov8m.pt')

results = model.train(
    data='deblur_coco.yaml',
    epochs=25,
    imgsz=640,
    batch=16,
    freeze=4,
    lr0=0.001,
    project='finetuning_deblur_stratified_data',
    name='yolov8m_frozen4_backbone',
    device=0 
)