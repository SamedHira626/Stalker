# Stalker

# The aim is to follow objects I trained

While I was working at [Open Zeka](https://openzeka.com), I built this car but before building that I worked on with [Jetson Inference](https://www.youtube.com/watch?v=QXIwdsyK7Rw&list=PL5B692fm6--uQRRDTPsJDp4o0xbzkoyf8&index=9), [DeepStream SDK](https://developer.nvidia.com/deepstream-sdk), [Hello AI World](https://www.youtube.com/watch?v=uvU8AXY1170&list=PL5B692fm6--uQRRDTPsJDp4o0xbzkoyf8), [Transfer Learning Toolkit](https://docs.nvidia.com/tao/tao-toolkit/text/overview.html) and [Jetbot](https://jetbot.org/master/)

# Materials:
Jetson Nano: https://openzeka.com/urun/nvidia-jetson-nano-developer-kit/

CSI Camera: https://openzeka.com/urun/raspberry-pi-kamera-v2/

# View of my car in Autodesk Fusion360:

First of all, I started this work by drawing a simple car in Fusion360.

![File_005](https://user-images.githubusercontent.com/42544569/132006355-30b4aa97-00c3-4744-8a74-a01753b42b16.jpeg)

# 3D printing:

![File_002](https://user-images.githubusercontent.com/42544569/132006708-fee7c41a-daa0-4378-ac10-11354b044a98.png)

# Final View

I mounted Jetson Nano, L298N motor driver and DC motors

![File_000](https://user-images.githubusercontent.com/42544569/132006371-54e729f5-9645-4d9a-994a-796f05c74173.png)

# KNOW-HOW
Hi, the following text is going to explain creating dataset, moving the car and how to build this car.

Firstly, I started to use [Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html) in order to create a scene like the figure below.

![Screenshot from 2021-08-06 12-47-27](https://user-images.githubusercontent.com/42544569/132004829-da659874-7d3d-4d34-b475-6225a2615112.png)

Secondly, I used [Synthetic Data Recorder](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/ext_omni_isaac_synthetic_utils_syntheticdata_recorder.html) for gathering image data with respect to their coordinates so that my dataset is created with their labels in npy file format. As the dataset was recorded as npy file, I wrote a pyhton script that converts npy file to txt file for yolo training. In the figure below, you can see how it looks like. By the way, I used [Domain randomization](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/sample_syntheticdata.html) to use texture component, light component, movement component etc. (You can see how I collected data using Domain Randomization in this link: https://youtu.be/1a3Q7aID_Ag). As a result, data collection with labels just got 30 minutes by the help of [Isaac Sim](https://docs.omniverse.nvidia.com/app_isaacsim/app_isaacsim/overview.html)

Also, you can see the rest of my data set: https://drive.google.com/drive/folders/1Kev6fsuPnR0Kro1vbTEKbixf1MmVFES5

![158](https://user-images.githubusercontent.com/42544569/132005462-a5aad6b3-e7a7-4dc3-bc1b-17d1e43db659.png)
![144](https://user-images.githubusercontent.com/42544569/132005470-59825197-50e2-4edb-ba12-698e95ad3650.png)

Thirdly, I trained YOLOv4-tiny on [Google Colab](https://colab.research.google.com), then converted my YOLOv4-tiny model to TensorRT by the following link (Just use Demo 5 in this repository): https://github.com/SamedHira626/tensorrt_demos  


Finally, clone this repository and just run stalker.py with python3:

>  python3 stalker.py



# Full Video Link is Avaliable: https://youtu.be/qP-J14y7uEY






