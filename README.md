# Motion-Deblurring
`Assignment-1 for COMP6001`

## Task-1
This repo is the working git repository. It includes the codes and the `gen_AI_logs.txt`. 

#### Ethical Considerations
- The data taken is from public domain and is publicaly available. The datasets used are the GoPRO dataset and the COCO validation set from 2017.
- The methods used with freezing 10 and 4 layers separately are done to mitigate intrinsic bias as much as possible.
- We try to prioritise information retention while trying to adapt to new domain(Deblurred Images)


## Task-2
The notebooks `Deblurring_Analysis_go_pro.ipynb` and `Deblurring_Analysis_coco.ipynb` contain all the code for the deblurring analysis and score comparisons. We calculate a `blur_score` using ratio of high frequencies to low frequencies. We also compare PSNR, SSIM, Blur Score Improvement post deblurring. We have created 3 buckets of images based off of the blur intensities. They are "low", "medium" and "high" blur image buckets.

**The 3 de-blurring algorithms we compare are:**
1. Wiener Restoration
2. Richardson Lucy
3. Google maxim deblur model (Trained on GoPRO)

Although we see the deep learning model perform best according to all metrics. We choose Wiener process due to its inference speed and computational cost. It provides a good middle ground between all metrics among all the 3 algorithms.

## Task-3
- The 2 scripts that correspond to this are `DETR_task_3_go_pro.py` and `YOLO_task_3_coco.py`. The output of these 2 scripts are in the folder `stat_output_logs`. 

- We can see a head to head comparison for YOLOv8 vs DETR on the goPRO dataset. In the `stat_output_logs/task_3_op_gopro` folder we have 2 files `detr_f1_score_comparison.png`, `yolo_f1_comparison.png`. This graph clearly shows how YOLOv8 is a better model than DETR for the specific task on the GoPRO dataset.

- Hence we choose YOLOv8 as our primary Object Detection model. We will Finetune the same as well. Metrics like mAP@50 and precision, recall are given in the report. We also have a notebook called `YOLOv8_eval_metrics.ipynb` which shows all the detailed metrics like per class AP and other metrics yolo tracks.  For more detailed metrics you can check the `metrics_log_train_and_val` folder. 

- These will be discussed more comprehensively and in detail in Task-4 and Task-5.

## Task-4
- The script `ssim_stratified_split_creator.py` contains the code for creating stratified splits based on blur intensity. This means it splits the data in such a way that train, test and val have equal samples that correspond to "low", "medium" and "high" blur intensities. 

- The dataset we used is the blurred version of the COCO_VAL_2017 provided to us for the assignment. The sharp dataset is the actual original COCO_VAL_2017 dataset. The de-blur version of the dataset is just a Wiener restored version of the dataset. The object detection annotations remain the same across all the datasets since the objects dont really change their location as such.

- The `ssim_stratified_split_creator.py` is designed to split the data into a YOLO friendly format in terms of both labels and the images. It follows the ultralytics format. 

- The split we follow is a 70-15-15 split. Since the dataset is a 5000 image dataset, we get 750 images for testing, 750 for validation and 3500 for training.

- The `finetune_yolo_with_blur.py` script provides the code for finetuning the YOLOv8 model on the de-blurred dataset. The `yaml` file for the same is `deblur_coco.yaml`. This file is just a pointer that tells the ultralytics fientuning pipeline where to look at for images and their labels. 

- Metrics for the finetuning in both the configurations(freezing 4 layers and freezing 10 layers) can be found in the `metrics_log_train_and_val` folder. This folder contains the training metrics along with the eval metrics on the test set.

## Task-5
- Once we have both the model weights in both the configurations, we can use the inbuilt validation function in YOLOv8 which allows us to calculate metrics on the test set(s) across sharp, blur and deblur datasets. There are 3 `yaml` files in the repo corresponding to each of these. They are `sharp_coco.yaml`, `blur_coco.yaml` and `deblur_coco.yaml`.

- Failure cases, root cause analysis and future scopes are explained in the report in detail.