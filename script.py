from ultralytics import YOLO
import cv2
import numpy as np

# TO DO
  # - add array as source
  # - disable irrelevant coco classes
  # - create and save masks

model = YOLO('yolov8n.pt')

source = 'input-images/room1.jpg'

results = model(source)

# View results
for r in results:
    orig_size = r.orig_shape
    height = orig_size[0]
    width = orig_size[1]
    print(height)
    print(width)

    boxes = r.boxes.numpy()
    xyxys = boxes.xyxy
    # for xyxy in xyxys:

    print(xyxys)  # print the Boxes object containing the detection bounding boxes