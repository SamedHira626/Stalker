# Stalker

# The aim is to follow objects I trained

# Materials:
Jetson Nano: https://openzeka.com/urun/nvidia-jetson-nano-developer-kit/

CSI Camera: https://openzeka.com/urun/raspberry-pi-kamera-v2/

# STL file of my car:

![File_005](https://user-images.githubusercontent.com/42544569/132006355-30b4aa97-00c3-4744-8a74-a01753b42b16.jpeg)

# 3D printing:

![File_002](https://user-images.githubusercontent.com/42544569/132006708-fee7c41a-daa0-4378-ac10-11354b044a98.png)

# Final View
![File_000](https://user-images.githubusercontent.com/42544569/132006371-54e729f5-9645-4d9a-994a-796f05c74173.png)

# KNOW-HOW
Hi, the following text is going to explain creating dataset, moving the car and how to build this car.

Firstly, I used Isaac Sim by starting to create a scene like figure below.

![Screenshot from 2021-08-06 12-47-27](https://user-images.githubusercontent.com/42544569/132004829-da659874-7d3d-4d34-b475-6225a2615112.png)

Secondly, I used Synthetic Data Recorder for gathering image data with respect to their coordinates so that my dataset is created with their labels in npy file format, as the dataset was recorded as npy file, I wrote a pyhton script that converts npy file to txt file for yolo. In the below figure, you can see how it looks like. By the way, I used Domain randomization to use texture component, light component, movement component etc. (You can see how I collected data using Domain Randomization in this link: https://youtu.be/1a3Q7aID_Ag)

![158](https://user-images.githubusercontent.com/42544569/132005462-a5aad6b3-e7a7-4dc3-bc1b-17d1e43db659.png)
![144](https://user-images.githubusercontent.com/42544569/132005470-59825197-50e2-4edb-ba12-698e95ad3650.png)

Thirdly, I converted my YOLOv4-tiny model to TensorRT by the following link (Just use Demo 5 in this repository): https://github.com/SamedHira626/tensorrt_demos  


Finally, clone this repository and just run stalker.py with python3:

>  python3 stalker.py



# Full Video Link is Avaliable: https://www.youtube.com/watch?v=bYLFatwzuRs






