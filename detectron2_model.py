from google.colab import drive

# Mount Google Drive
drive.mount('/content/gdrive')


import detectron2
from detectron2.data.datasets import register_coco_instances
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader

# Step 1: Convert JSON annotations to COCO format

import json
from collections import defaultdict

# Load the JSON data from the file
file_path = '/train_labels_100.json'
with open(file_path, 'r') as f:
    json_data = json.load(f)

# Create a default dictionary to store annotations for each image
coco_annotations = defaultdict(list)

# Function to convert xmin, ymin, xmax, ymax to bbox format
def convert_to_bbox(xmin, ymin, xmax, ymax, width, height):
    x, y, w, h = float(xmin), float(ymin), float(xmax) - float(xmin), float(ymax) - float(ymin)
    return [x * width, y * height, w * width, h * height]

# Loop through each JSON entry and convert to COCO format
for entry in json_data:
    image_id = entry["filename"]
    width, height = entry["width"], entry["height"]
    bbox = convert_to_bbox(entry["xmin"], entry["ymin"], entry["xmax"], entry["ymax"], width, height)
    category_id = entry["class"]

    coco_entry = {
        "image_id": image_id,
        "category_id": category_id,
        "bbox": bbox,
        "area": bbox[2] * bbox[3],
        "iscrowd": 0,
        "id": entry["FIELD1"]
    }

    coco_annotations[image_id].append(coco_entry)

# Create the final COCO format dictionary
coco_data = {
    "images": [{"file_name": image_id, "height": height, "width": width, "id": idx} for idx, image_id in enumerate(coco_annotations.keys())],
    "annotations": [entry for entries in coco_annotations.values() for entry in entries],
    "categories": [{"id": idx, "name": category_id} for idx, category_id in enumerate(set(entry["category_id"] for entries in coco_annotations.values() for entry in entries))]
}

# Save the COCO format data to a file
with open("coco_annotations.json", "w") as f:
    json.dump(coco_data, f)


# Step 2: Register the dataset in Detectron2
path_to_coco_annotations = '/content/coco_annotations.json'
path_to_images_directory = ''                                       # missing image directory

register_coco_instances("custom_dataset", {}, path_to_coco_annotations, path_to_images_directory)

# Step 3: Define the model and configuration
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("custom_dataset",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 4
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 10000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Number of classes (excluding background)

# Step 4: Train the model
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Step 5: Evaluate the model
evaluator = COCOEvaluator("custom_dataset", cfg, False, output_dir=cfg.OUTPUT_DIR)
val_loader = build_detection_test_loader(cfg, "custom_dataset")
inference_on_dataset(trainer.model, val_loader, evaluator)

# Step 6: Use the trained model for inference
from detectron2.engine import DefaultPredictor

predictor = DefaultPredictor(cfg)
image_path = ""                                 # Add the directory path to the image that you want to use
outputs = predictor(image_path)
print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)


'''
REPORT DATE: 24th July 2023

Today, I worked on the detectron2 code using Google Colab, aiming to build my own object detection model. My starting point was a CSV file containing over 100,000 objects, 
which required conversion to a JSON file. To do this task, I created a Python script called 'csv-to-json.py', which can be found in this repository.

Throughout the day, I dedicated time to studying online tutorials that covered the fundamentals of detectron2, equipping myself with the necessary knowledge to create my model. 
For the initial stages, I focused on a subset of 100 JSON annotations to streamline the process of building the model with annotated steps to keep track of my steps.

However, during my research, I discovered that I need to obtain the corresponding image files for these JSON annotations before proceeding with the training phase. 
This step is crucial to ensure the model's effectiveness and accuracy in detecting objects.

Moving forward, I plan to gather the required image files, and I am excited to delve deeper into the process of training and optimizing my object detection model using detectron2.
'''
