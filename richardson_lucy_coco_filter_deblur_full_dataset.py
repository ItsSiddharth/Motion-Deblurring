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

import numpy as np
from skimage import restoration
from PIL import Image
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

import cv2
from tqdm import tqdm
import os

def calculate_blur_score_fft(input_data, radius=30):
    if isinstance(input_data, str):
        img = cv2.imread(input_data, 0)
    else:
        if len(input_data.shape) == 3:
                # Handle float (0-1) vs uint8 (0-255)
                if input_data.dtype != np.uint8:
                    input_data = (input_data * 255).astype(np.uint8)
                img = cv2.cvtColor(input_data, cv2.COLOR_RGB2GRAY)
        else:
            img = input_data
    rows, cols = img.shape
    crow, ccol = rows // 2 , cols // 2
    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft) # this function helps shift all the zero freq points to centre
    magnitude_spectrum = np.abs(dft_shift)
    mask = np.zeros((rows, cols), np.uint8)
    cv2.circle(mask, (ccol, crow), radius, 1, -1)
    low_freq_area = magnitude_spectrum * mask
    high_freq_area = magnitude_spectrum * (1 - mask)
    low_energy = np.sum(low_freq_area)
    high_energy = np.sum(high_freq_area)
    ratio = high_energy / low_energy
    return ratio

def richardson_lucy_filtering_each_channel(blur_image_path, iterations=30, plot=False):
    original_blur_score = calculate_blur_score_fft(blur_image_path)
    blur_image = Image.open(blur_image_path)
    blur_image = np.array(blur_image) / 255.0
    psf = np.ones((5, 5)) / 25
    
    deconvolved_img = np.zeros_like(blur_image)
    
    for i in range(3):
        # Richardson-Lucy deconvolution
        deconvolved_img[:, :, i] = restoration.richardson_lucy(
            blur_image[:, :, i], 
            psf, 
            num_iter=iterations
        )
    deconvolved_img = np.clip(deconvolved_img, 0, 1)
    filtered_blur_score = calculate_blur_score_fft(deconvolved_img)

    if plot:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        ax = axes.ravel()

        ax[0].imshow(blur_image)
        ax[0].set_title(f"Original Score: {original_blur_score:.2f}")
        ax[0].axis('off')

        ax[1].imshow(deconvolved_img)
        ax[1].set_title(f"Richardson-Lucy ({iterations} iter) Score: {filtered_blur_score:.2f}")
        ax[1].axis('off')

        plt.tight_layout()
        plt.show()

    return deconvolved_img, original_blur_score, filtered_blur_score

skipped = 0
image_and_blur_score = []
list_of_files = os.listdir("/home/nam/projects/sid/Motion-Deblurring/coco_val_data_and_annots/blurred_images_coco")
already_deblurred = os.listdir("/home/nam/projects/sid/Motion-Deblurring/coco_val_data_and_annots/richardson_lucy_deblur_coco")
remaining_files = [item for item in list_of_files if item not in already_deblurred]
for image_file in tqdm(remaining_files):
    try:
        if image_file not in already_deblurred:
            full_path = os.path.join("/home/nam/projects/sid/Motion-Deblurring/coco_val_data_and_annots/blurred_images_coco", image_file)
            full_path_sharp = os.path.join("/home/nam/projects/sid/Motion-Deblurring/coco_val_data_and_annots/sharp_images_coco", image_file)
            sharp_image = Image.open(full_path_sharp)
            sharp_image = np.array(sharp_image) / 255.0
            deblurred_img, original_score, filtered_blur_score = richardson_lucy_filtering_each_channel(full_path)
            deblurred_img_uint8 = (deblurred_img * 255).astype(np.uint8)
            deblurred_img_pil = Image.fromarray(deblurred_img_uint8)
            deblurred_img_pil.save(f"coco_val_data_and_annots/richardson_lucy_deblur_coco/{image_file}")
            if sharp_image.shape != deblurred_img.shape:
                # Resize deblurred to match sharp (skimage uses H,W,C, cv2 uses W,H)
                h, w = sharp_image.shape[:2]
                deblurred_img = cv2.resize(deblurred_img, (w, h))
            psnr_score = round(psnr(sharp_image, deblurred_img, data_range=1.0), 3)
            ssim_score = round(ssim(sharp_image, deblurred_img, data_range=1.0, channel_axis=-1), 3)
            image_and_blur_score.append([full_path, original_score, filtered_blur_score, psnr_score, ssim_score])
            sorted_blur_scores = sorted(image_and_blur_score, key=lambda x: x[2], reverse=True)
    except:
        skipped += 1
        continue
original_blur_scores_for_plot = [score[1] for score in sorted_blur_scores]
blur_scores_for_plot = [score[2] for score in sorted_blur_scores]
psnr_score_for_plot = [score[3] for score in sorted_blur_scores]
ssim_score_for_plot = [score[4] for score in sorted_blur_scores]

x = np.arange(len(original_blur_scores_for_plot))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 5))
rects1 = ax.bar(x - width/2, original_blur_scores_for_plot, width, label='Original', color='skyblue')
rects2 = ax.bar(x + width/2, blur_scores_for_plot, width, label='Filtered', color='salmon')
ax.set_ylabel("Blur Scores")
ax.set_title("Comparison of Blur Scores - Richardson Lucy")
ax.legend()
ax.set_xticks([])

plt.tight_layout()
plt.show()

differences = np.array(blur_scores_for_plot) - np.array(original_blur_scores_for_plot)
avg_increase = np.mean(differences)

avg_original = np.mean(np.array(original_blur_scores_for_plot))
percent_increase = (avg_increase / avg_original) * 100

print(f"--- Blur Score Statistics ---")
print(f"Average Original Score:  {avg_original:.2f}")
print(f"Average Filtered Score:  {np.mean(blur_scores_for_plot):.2f}")
print(f"Average Absolute Increase: {avg_increase:.2f}")
print(f"Average Percentage Increase: {percent_increase:.2f}%")
print(f"Average PSNR score: {np.mean(psnr_score_for_plot)}")
print(f"Average SSIM score: {np.mean(ssim_score_for_plot)}")

best_improvement = np.max(differences)
print(f"Highest single image improvement: {best_improvement:.2f}")