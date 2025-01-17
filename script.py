from ultralytics import YOLO
from ultralytics import settings
import os
import shutil
import cv2
import numpy as np
import pathlib
from pathlib import Path

relevant_classes = ['chair', 'couch', 'bed', 'dining table']

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

def deleteIrrelevant(directory):
  # delete irrelevant cropped items
  for p, folders, files in os.walk(directory):
    for folder_name in folders:
      if folder_name not in relevant_classes:
        path_remove = os.path.join(directory, folder_name)
        shutil.rmtree(path_remove)
  
  # delete result images 
  parent_dir = getPathToSave()
  for filename in os.scandir(parent_dir):
    if filename.is_file():
      os.remove(filename.path)

def saveMasks(masks, img_id, path, result):
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

  # save individual cropped pieces of furniture 
  result.save_crop(final_path)

  # save the original image to this directory
  orig_img = result.orig_img
  name = 'original.jpg'
  img_path = os.path.join(final_path, name)
  cv2.imwrite(img_path, orig_img)

  # delete irrelevant stuff
  deleteIrrelevant(final_path)

def findRelevantClasses(results):
  # find values/ids of relevant classes
  relevant_values = []
  dictionary = results[0].names # returns value:className pairs of all classes

  for val, class_name in dictionary.items():
    if class_name in relevant_classes:
      relevant_values.append(val)

  return relevant_values

def analyzeResults(results, path, relevant_values):
  img_id = 0

  for r in results:
    img_id += 1
    masks = []

    # filter only relevant boxes 
    boxes = r.boxes.numpy()
    for box in boxes:
      if box.cls in relevant_values:

        # get start and end points of the bounding box
        xyxy = box.xyxy
        xyxy = xyxy[0] # getting rid of redundant double array
        start_point = (int(xyxy[0]), int(xyxy[1]))
        end_point = (int(xyxy[2]), int(xyxy[3]))

        # create a mask
        mask = createMask(r, start_point, end_point)
        masks.append(mask)

    # save masks of the image    
    saveMasks(masks, img_id, path, r)

def main():

  # load a pretrained model
  model = YOLO('yolov8n.pt')

  # name of the folder with input images
  source = 'input-images'

  # run the model on the input images and save results
  results = model(source, save = True)
  
  path = getPathToSave()

  relevant_values = findRelevantClasses(results)

  analyzeResults(results, path, relevant_values)

if __name__ == '__main__':
  main()
