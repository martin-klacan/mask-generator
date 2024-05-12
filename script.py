from ultralytics import YOLO
from ultralytics import settings
import os
import cv2
import numpy as np
import pathlib
from pathlib import Path

# TO DO
  # - disable irrelevant coco classes

def createMask(result, start_point, end_point):
  orig_size = result.orig_shape
  height = orig_size[0]
  width = orig_size[1]

  # create a black image
  img = np.zeros((height, width, 3), dtype = np.uint8)
    
  color = (255, 255, 255)
  thickness = -1 # value -1 will fill the entire shape

  # add white rectangle to the black image
  img = cv2.rectangle(img, start_point, end_point, color, thickness)

  return img

def getPathToSave():
  runs_dir = settings['runs_dir']
  temp_path = runs_dir + '\\detect\\'

  # find the latest modified/created directory
  newest_dir = max(pathlib.Path(temp_path).glob('*/'), key=os.path.getmtime)
  return newest_dir

def saveMasks(masks, img_id, path):
  # make new directory for image masks
  parent_directory = path
  new_directory = 'masks-room' + str(img_id)
  final_path = os.path.join(parent_directory, new_directory)
  os.mkdir(final_path)
  
  # save masks into the new directory
  i = 0
  for mask in masks:
    i += 1
    name = 'mask' + str(i) + '.png'
    img_path = os.path.join(final_path, name)
    cv2.imwrite(img_path, mask)

# def getImages():

  # # add all input images to a list
  # temp_list = []
  # folder_dir = 'input-images'
  # images = Path(folder_dir).glob('*.jpg')
  # for image in images:
  #   temp_list.append(image)
  # return temp_list

def analyzeResults(results, path):
  img_id = 0
  for r in results:
    img_id += 1

    # get coordinates of all bounding boxes
    boxes = r.boxes.numpy()
    xyxys = boxes.xyxy

    masks = []
    # get start and end points of all bounding boxes and create masks
    for xyxy in xyxys:
      start_point = (int(xyxy[0]), int(xyxy[1]))
      end_point = (int(xyxy[2]), int(xyxy[3]))

      # create a mask
      mask = createMask(r, start_point, end_point)
      masks.append(mask)

    # save masks of the image
    saveMasks(masks, img_id, path)

def main():

  model = YOLO('yolov8n.pt')

  # source = getImages()
  # name of the folder with input images
  source = 'input-images'

  results = model(source, save=True)
  
  path = getPathToSave()

  analyzeResults(results, path)

if __name__ == '__main__':
  main()






