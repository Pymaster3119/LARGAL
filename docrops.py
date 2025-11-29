import os
import shutil
import json
import tqdm
import cv2 as opencv
import numpy as np
import pickle

#create .txt files for YOLO annotations
ann_coco = ["RadioGalaxyNET_Dataset/data/RadioGalaxyNET/annotations/train.json","RadioGalaxyNET_Dataset/data/RadioGalaxyNET/annotations/test.json", "RadioGalaxyNET_Dataset/data/RadioGalaxyNET/annotations/val.json"]
ann_yolo = ["traincubicfit.pkl", "testcubicfit.pkl", "valcubicfit.pkl"]
things = ["train", "test", "val"]

for coco_ann, yolo_ann, thing in zip(ann_coco, ann_yolo, things):
    out_x = []
    out_y = []
    out_areas = []
    with open(coco_ann, 'r') as f:
        coco_data = json.load(f)
    for img_info in tqdm.tqdm(coco_data['images']):
        img_id = img_info['id']
        annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == img_id]
        # Open this image with opencv
        img_path = os.path.join(f"RadioGalaxyNET_Dataset/data/RadioGalaxyNET/{thing}", img_info['file_name'])
        img = opencv.imread(img_path)

        for anno in annotations:
            # Convert COCO bbox [x,y,width,height] to YOLO [class_id, x_center, y_center, width, height]
            class_id = anno['category_id']
            x_center = (anno['bbox'][0] + anno['bbox'][2] / 2)
            y_center = (anno['bbox'][1] + anno['bbox'][3] / 2)
            width = anno['bbox'][2]
            height = anno['bbox'][3]

            #Crop out a rectangle, scale it to up to double and pad to 64x64 if needed, and then save to list 
            x1 = int(anno['bbox'][0])
            y1 = int(anno['bbox'][1])
            x2 = int(anno['bbox'][0] + anno['bbox'][2])
            y2 = int(anno['bbox'][1] + anno['bbox'][3])
            h, w = img.shape[:2]
            x1 = max(0, min(x1, w))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h))
            y2 = max(0, min(y2, h))
            crop = img[y1:y2, x1:x2].copy()
            # Best fit the image to 64x64 with padding
            crop_h, crop_w = crop.shape[:2]
            scale = min(64 / crop_w, 64 / crop_h)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            resized_crop = opencv.resize(crop, (new_w, new_h), interpolation=opencv.INTER_CUBIC)
            padded_crop = np.zeros((64, 64, 3), dtype=resized_crop.dtype)
            padded_crop[(64 - new_h) // 2:(64 - new_h) // 2 + new_h, (64 - new_w) // 2:(64 - new_w) // 2 + new_w] = resized_crop
            area = anno['area']
            crop = padded_crop#opencv.resize(crop, (64, 64), interpolation=opencv.INTER_CUBIC)
            
            print(area)
            out_x.append(crop)
            out_y.append(class_id)
            out_areas.append(area)
    #Convert to numpy and save with pickle
    out_x = np.array(out_x)
    out_y = np.array(out_y)
    out_areas = np.array(out_areas)
    print(out_x.shape, out_y.shape, out_areas.shape)
    with open(yolo_ann, 'wb') as f:
        pickle.dump((out_x, out_y), f)
    with open(yolo_ann.replace('.pkl', '_areas.pkl'), 'wb') as f:
        pickle.dump((out_x, out_y, out_areas), f)
        
        