from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# TO DO
  # - add array as source
  # - disable irrelevant coco classes
  # - create and save masks

def createMask(result, start_point, end_point):
  orig_size = result.orig_shape
  height = orig_size[0]
  width = orig_size[1]

  # create a black image
  img = np.zeros((height, width, 3), dtype = np.uint8)
    
  color = (255, 255, 255)
  thickness = -1 # will fill the entire shape

  img = cv2.rectangle(img, start_point, end_point, color, thickness)

  cv2.imshow('image', img) 

def main():

  model = YOLO('yolov8n.pt')
  source = []

  folder_dir = 'input-images'
  images = Path(folder_dir).glob('*.jpg')
  for image in images:
    source.append(image)

  results = model(source, save=True)

  # View results
  for r in results:
    boxes = r.boxes.numpy()
    xyxys = boxes.xyxy

    # get start and end points of all boxes and create masks
    for xyxy in xyxys:
      start_point = (int(xyxy[0]), int(xyxy[1]))
      end_point = (int(xyxy[2]), int(xyxy[3]))
      print(start_point)
      print(end_point)
      createMask(r, start_point, end_point)

    print(xyxys)  # print the Boxes object containing the detection bounding boxes

if __name__ == '__main__':
  main()






