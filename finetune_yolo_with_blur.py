from ultralytics import YOLO

model = YOLO('yolov8m.pt')

results = model.train(
    data='deblur_coco.yaml',
    epochs=25,
    imgsz=640,
    batch=16,
    freeze=10,
    lr0=0.001,
    project='finetuning_deblur_stratified_data',
    name='yolov8m_frozen10_backbone',
    device=0 
)