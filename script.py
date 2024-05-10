from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

# TO DO
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

  return img

# def saveMasks(masks):
  # TO DO
    # - get path that the model uses to save results
    # - create a folder there for each image
    # - save corresponiding masks in the corresponding folder

def getImages():

  # add all input images to a list
  tempList = []
  folder_dir = 'input-images'
  images = Path(folder_dir).glob('*.jpg')
  for image in images:
    tempList.append(image)
  return tempList

def main():

  model = YOLO('yolov8n.pt')

  source = getImages()

  results = model(source)
  print(results)

  for r in results:
    # get coordinates of all bounding boxes
    boxes = r.boxes.numpy()
    xyxys = boxes.xyxy

    masks = []
    # get start and end points of all bounding boxes and create masks
    for xyxy in xyxys:
      start_point = (int(xyxy[0]), int(xyxy[1]))
      end_point = (int(xyxy[2]), int(xyxy[3]))
      mask = createMask(r, start_point, end_point)
      masks.append(mask)

    # print(xyxys)  # print the Boxes object containing the detection bounding boxes
    saveMasks(masks)

if __name__ == '__main__':
  main()






