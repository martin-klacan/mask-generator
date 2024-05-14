# Mask Generator

This project acts as a tool for organizations in the furniture industry. It solves an object detection problem, by detecting various furniture items in an image of a room and creates a black-and-white mask for each item. These can further be used for example to furnish the room according to the preferences to the user. 

## Description

In the project we make use of the extensive Ultralytics model for object detection which is pretrained on coco8 dataset. This allows us to detect and classify furniture items with very high confidence and consistency. After that the script creates black masks with white rectangle inside of them corresponging to the piece of furniture. These masks are the main output of this project and so they are saved in a folder ``` \runs\detect\predict<#>\masks-room<#> ``` where "#" is autmatically assigned. The script works also for multiple room images at ones, which are saved in the  ```\input-images``` folder. In that case the masks are saved in the corresponding folders for each image.
The script focuses on just the relevant classes for the user, which are specified in a global variable (default being ```chair, couch, bed, dining table```), all other classes are being filtered out.
For convenience, 2 additional things are saved in the ```masks-room<#>``` folder - the actual cropped images of the furniture items and the original image.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

Before running the script there are a few steps you need to take:

* Make sure to have pyhton installed
```install python3```
* You also need a package installer for python, for example pip, which may be preinstalled with python3
* Next you need to install pytorch, you can follow the guide based on your OS: https://pytorch.org/get-started/locally/ and install for CPU platform. For windows:
```pip install torch torchvision torchaudio```
WARNING: The following error may occur when having the newest version of pytorch, when installing Ultralytics in the next step: 
```"The specified module could not be found. Error loading "C:\Users... ...\Python312\site-packages\torch\lib\shm.dll" or one of its dependencies."```
This bug has not yet been resolved my knowledge and the only way to fix it is to install an older version of pytorch, namely 2.2.2, using this command for windows:
```pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cpu```
* The most important step is installing Ultralytics, from which we are using the model, with a simple command:
```pip install ultralytics```
Note: you don't need to clone the ultralytics repo, everything that's necessary will be downloaded during execution


### Installation

To run this script, simply clone the repo from GitHub

```$ git clone https://github.com/martin-klacan/mask-generator.git```

and run the script with a command
```python3 script.py```

## Usage

To use this script, make sure to have some images stored in the ```input-images``` folder or just use the 3 images that are already stored there 

After running the script the results will be saved in the folder ```\runs\detect\predict<#>\masks-room<#>```. There are already results from 2 example runs saved in ```predict``` and ```predict2``` folders. Further runs will naturally have ids from 3 onwards or if you delete any, the smallest id available. 

## Approach

My approach during the developement of this project was to first study the model thorougly in the Ultralyitcs extensive documentation, including videos and examples of usage. After having sufficient understanding of the model I started experimenting with its features and capabilities, with the intention to find the simplest and most clever solution to this problem. I coded with the intention for the code to be readable and easily understandable for another reader or a team member, I aimed to use descriptive variable names and make use of encapsulated functions which have clear purpose, input and output. I also focused on keeping the main() as clean and short as possible and made use of comments wherever needed.   

## Author

Martin Klačan


