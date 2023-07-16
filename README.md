# Counting People
Analyzed a dataset consisting of videos recorded at a bus entrance in China. Annotated the images in video frames and trained a model to count the number of people passing through a specified scene using TensorFlow Object Detection.

---
## **Explaination**
The videos are annotated frame by frame, each second, and then annotated and split using Roboflow. I used Yolov5 to detect people in each image (frame). In the following, I will explain it in detail.

After going to the main directory of the project in google colab, we should clone the Yolov5
```ruby
# clone YOLOv5
!git clone https://github.com/ultralytics/yolov5  # clone repo
%cd yolov5
%pip install -qr requirements.txt # install dependencies
%pip install -q roboflow
```
After that we should add torch and os libraries
```ruby
import torch
import os
from IPython.display import Image, clear_output  # to display images
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")
```
Then, we should import our dataset from the roboflow
```ruby
from roboflow import Roboflow
rf = Roboflow(model_format="yolov5", notebook="ultralytics")
```
```ruby
os.environ["DATASET_DIRECTORY"] = "/content/drive/MyDrive/AI_Project"
```
```ruby
from roboflow import Roboflow
rf = Roboflow(api_key="6MZs0IEOrn1yaUepdoxH")
project = rf.workspace("ca-foscari-university-of-venice").project("countpeople")
dataset = project.version(14).download("yolov5")
```
By applying the following code we consider human detection in such a way that the number of epochs, number of batches and image size. 
```ruby
!python train.py --img 416 --batch 16 --epochs 25 --data {'/content/drive/MyDrive/AI_Project/CountPeople-14'}/data.yaml --weights yolov5s.pt --cache
```
to consider the objects as human we set **--conf** to **0.7** to say if the probability was more than 70% it should be human. 
```ruby
!python /content/drive/MyDrive/AI_Project/yolov5/detect.py --weights /content/drive/MyDrive/AI_Project/yolov5/runs/train/exp/weights/best.pt --img 416 --conf 0.7 --source {'/content/drive/MyDrive/Pellilo/Main/CountPeople-14'}/test/images
```
To read all the images from the directory we can use
```ruby
import cv2 as cv
import glob
import torch
import pandas as pd
import numpy as np

data = []
model = torch.hub.load('/content/drive/MyDrive/AI_Project/yolov5','custom', path='/content/drive/MyDrive/AI_Project/yolov5/runs/train/exp/weights/best.pt', source='local')

files = glob.glob("/content/drive/MyDrive/AI_Project/yolov5/runs/detect/exp/*.jpg")

for myFile in files:
    img = cv.imread(myFile)
    
    results = model(img)
    #results.print() 
    display(Image(filename=myFile))
    data.append(results)

df = pd.DataFrame(data)
```
```ruby
df.to_csv("df.csv", index=False)
```
```ruby
# Load the csv file into a pandas dataframe
df = pd.read_csv('df.csv')

# Split the strings in each cell into a list of strings
split_strings = df[df.columns[0]].apply(lambda x: x.split(" "))

# Convert the list of strings into a dataframe
result = pd.DataFrame(split_strings.tolist(), columns=[f"Column {i}" for i in range(1, len(max(split_strings, key=len)) + 1)])
```
```ruby
result.columns = ['Column_1', 'Column_2','Column_3', 'Column_4','Column_5', 'Column_6','Column_7', 'Column_8','Column_9', 'Column_10','Column_11', 'Column_12','Column_13', 'Column_14','Column_15', 'Column_16','Column_17', 'Column_18','Column_19']
```
```ruby
result['Column_4'] = pd.to_numeric(result['Column_4'], errors='coerce')
People = result['Column_4'].sum()

print("Total number of peoples in the all images is:", People)
```
