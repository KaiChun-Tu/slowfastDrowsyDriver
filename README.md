# Development of nighttime fatigue driving detection technology using infrared images
## Inroduction
This project discusses the nighttime IR image fatigue driving detection based on slowfast, in the following will describe how this project work, what and how dataset is used.

## Abstract
Traffic safety accidents are one of the top ten causes of death, with fatigue
driving accounting for a large proportion of them. Fatigue can cause reduced
concentration and reaction speed, and there are many occupations that often
drive at night because of their occupational needs, for example, long-haul
truck drivers, cab drivers, and so on are high-risk groups.
Therefore, to avoid accidents caused by fatigue driving, this paper proposes a real-time driving fatigue monitoring system, which uses a car recorder
image to capture the driving image, uses a deep learning model to capture the
driver’s head image and builds an abnormal event detection model to determine whether the driver has abnormal behavior due to fatigue and gives a
reminder to avoid accidents.
Therefore, this paper collects a series of IR images in the car at night and
annotates them according to the format of Kinectis data set.

## Installation
If you encounter any problems during installation, you can consult the origin slowfast. https://github.com/facebookresearch/SlowFast
### Requirments
* Python >= 3.8  
* Numpy  
* PyTorch >= 1.3  
* fvcore: pip install 'git+https://github.com/facebookresearch/fvcore'  
* torchvision that matches the PyTorch installation. You can install them together at pytorch.org to make sure of this.  
* simplejson: pip install simplejson  
* GCC >= 4.9  
* PyAV: conda install av -c conda-forge  
* ffmpeg (4.0 is prefereed, will be installed along with PyAV)  
* PyYaml: (will be installed along with fvcore)  
* tqdm: (will be installed along with fvcore)  
* iopath: pip install -U iopath or conda install -c iopath iopath  
* psutil: pip install psutil  
* OpenCV: pip install opencv-python  
* torchvision: pip install torchvision or conda install torchvision -c pytorch  
* tensorboard: pip install tensorboard  
* moviepy: (optional, for visualizing video on tensorboard) conda install -c conda-forge moviepy or pip install moviepy  
* PyTorchVideo: pip install pytorchvideo  
* Detectron2:  
* FairScale: pip install 'git+https://github.com/facebookresearch/fairscale'  

## Build
    git clone https://github.com/KaiChun-Tu/-Development-of-nighttime-fatigue-driving-detection-technology-using-infrared-images.git
    cd slowfastDrowsyDriver
    python setup.py build develop
    
# Dataset
## format 
The dataset is annotates in Kinetics format.  
Each dataset has two folders.  
* CSV folder: Contains three csv files, namely train, val, and test, which record the absolute path and category of each data.
* Clip folder: a folder that stores each data, each folder contains eight images and video files of the data
## Dataset Name  
Our night dataset  
* CSV folder：new_night  
* Clip folder：clip  
Our dynamic night dataset  
* CSV folder：dynamicNight  
* Clip folder：dynamicNight_clip  
