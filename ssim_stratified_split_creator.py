import os
import cv2
import shutil
import numpy as np
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from scipy.signal import wiener

def apply_wiener(image_path, mysize=None, noise=None):
    img = cv2.imread(image_path, 0)
    img_float = img.astype(np.float64) / 255.0
    restored = wiener(img_float, mysize=mysize, noise=noise)
    return (np.clip(restored, 0, 1) * 255).astype(np.uint8)

def setup_stratified_dataset(blur_dir, sharp_dir, label_dir, output_root):
    domains = ['sharp', 'blur', 'deblur']
    splits = ['train', 'val', 'test']
    
    for d in domains:
        for s in splits:
            os.makedirs(os.path.join(output_root, d, s, 'images'), exist_ok=True)
            os.makedirs(os.path.join(output_root, d, s, 'labels'), exist_ok=True)

    files = [f for f in os.listdir(blur_dir) if f.endswith(('.jpg', '.png'))]
    data_registry = []

    print("Step 1: Calculating Degradation (SSIM) and Deblurring...")
    for f in tqdm(files):
        b_path = os.path.join(blur_dir, f)
        s_path = os.path.join(sharp_dir, f)
        l_path = os.path.join(label_dir, os.path.splitext(f)[0] + ".txt")

        if not os.path.exists(s_path) or not os.path.exists(l_path):
            continue

        img_b = cv2.imread(b_path, 0)
        img_s = cv2.imread(s_path, 0)
        score = ssim(img_b, img_s)
        
        data_registry.append({'file': f, 'ssim': score, 'label': l_path})

    data_registry.sort(key=lambda x: x['ssim'], reverse=True)
    n = len(data_registry)
    
    tiers = {
        'Low_Blur': data_registry[:n//3],
        'Mid_Blur': data_registry[n//3 : 2*n//3],
        'High_Blur': data_registry[2*n//3:]
    }

    print("Step 2: Copying files and generating Deblurred domain...")
    for tier_name, members in tiers.items():
        np.random.seed(42)
        np.random.shuffle(members)
        
        # 70-15-15 Split
        t_idx = int(0.7 * len(members))
        v_idx = int(0.85 * len(members))
        
        split_map = {
            'train': members[:t_idx],
            'val': members[t_idx:v_idx],
            'test': members[v_idx:]
        }

        for split, items in split_map.items():
            for item in items:
                fname = item['file']
                
                shutil.copy(os.path.join(sharp_dir, fname), os.path.join(output_root, 'sharp', split, 'images', fname))
                shutil.copy(item['label'], os.path.join(output_root, 'sharp', split, 'labels', fname.replace('.jpg','.txt').replace('.png','.txt')))

                shutil.copy(os.path.join(blur_dir, fname), os.path.join(output_root, 'blur', split, 'images', fname))
                shutil.copy(item['label'], os.path.join(output_root, 'blur', split, 'labels', fname.replace('.jpg','.txt').replace('.png','.txt')))

                deblurred_img = apply_wiener(os.path.join(blur_dir, fname), mysize=5, noise=0.01)
                cv2.imwrite(os.path.join(output_root, 'deblur', split, 'images', fname), deblurred_img)
                shutil.copy(item['label'], os.path.join(output_root, 'deblur', split, 'labels', fname.replace('.jpg','.txt').replace('.png','.txt')))

    print(f"Dataset construction complete at {output_root}")

setup_stratified_dataset(
    blur_dir='/home/nam/projects/sid/Motion-Deblurring/coco_val_data_and_annots/blurred_images_coco',
    sharp_dir='/home/nam/projects/sid/Motion-Deblurring/coco_val_data_and_annots/sharp_images_coco',
    label_dir='/home/nam/projects/sid/Motion-Deblurring/coco_val_data_and_annots/yolo_format_annots',
    output_root='/home/nam/projects/sid/Motion-Deblurring/coco_val_data_and_annots/COCO_Restoration_Study'
)