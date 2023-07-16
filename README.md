# Counting People
Analyzed a dataset consisting of videos recorded at a bus entrance in China. Annotated the images in video frames and trained a model to count the number of people passing through a specified scene using TensorFlow Object Detection.

---
## **Explaination**
The videos are annotated frame by frame, each second, and then annotated and split using Roboflow. I used PyTorch to detect people in each image (frame). In the following, I will explain it in detail.
