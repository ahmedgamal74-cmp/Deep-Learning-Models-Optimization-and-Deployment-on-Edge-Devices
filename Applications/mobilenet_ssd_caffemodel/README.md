# Multi Object Tracking Using MobileNet SSD 

Implementation of Multi Object Tracking using a pretrained MobileNet SSD with dlib library and OpenCV in Python.

## Multi Object Tracking:
Multiple object tracking is the task of tracking more than one object in the video. Here, the algorithm assigns a unique variable to each of the objects that are detected in the video frame. Subsequently, it identifies and tracks all these multiple objects in consecutive/upcoming frames of the video.

## SSD MobileNet Architecture:
The SSD architecture is a single convolution network that learns to predict bounding box locations and classify these locations in one pass. Hence, SSD can be trained end-to-end. The SSD network consists of base architecture (MobileNet in this case) followed by several convolution layers:

![](images/ssd_architecture.png)

By using SSD, we only need to take one single shot to detect multiple objects within the image, while regional proposal network (RPN) based approaches such as R-CNN series that need two shots, one for generating region proposals, one for detecting the object of each proposal. Thus, SSD is much faster compared with two-shot RPN-based approaches.


## Output

![](images/output.gif)


## Requirements :

- dlib
- opencv-python
- imutils

## Usage :

- Clone this Repository
```
git clone https://github.com/ayanbag/Multi_Object_Tracking_with_MobileNetSSD.git
cd Multi_Object_Tracking_with_MobileNetSSD
```
Then run the following command to install the required dependencies.
```
pip install -r requirements.txt
```

- Now excute the following command :

```
python multi_object_tracking.py -i <path-to-input>
```
python multi_object_tracking.py -i <D:\yourfilelocation\input\aeroplane.mp4>


**Note:** Our script processes the following command line arguments at runtime:

- `--input` or `-i` : The path to the input video file. We’ll perform multi-object tracking with dlib on this video.
- `--confidence` or `-c` : An optional override for the object detection confidence threshold of 0.2 . This value represents the minimum probability to filter weak detections from the object detector.

# MobileNet-SSD Object Detection and MOSSE Tracking

This project performs real-time object detection using the MobileNet-SSD deep learning model, followed by object tracking using the MOSSE tracker. It works on both video files and webcam streams, saving the processed video with bounding boxes and labels.

## Features

- Object detection using pre-trained MobileNet-SSD (Caffe)
- Object tracking with OpenCV's MOSSE tracker
- Webcam or video file input support
- Real-time display and output video saving
- FPS calculation for performance monitoring

## Dependencies

- Python 3.x
- OpenCV (`opencv-python`, `opencv-contrib-python`)
- imutils
- numpy

Install dependencies using pip:

```bash
pip install opencv-python opencv-contrib-python imutils numpy
```

## Model Files

Ensure the following model files are downloaded and placed in the `mobilenet_ssd/` directory:

- `MobileNetSSD_deploy.prototxt`
- `MobileNetSSD_deploy.caffemodel`

## Usage

### 1. Webcam Mode (Default)

```bash
python object_tracking.py
```

### 2. Video File Input

```bash
python object_tracking.py --input path/to/video.mp4
```

### Optional Arguments

- `--input`: Path to input video file (leave empty for webcam)
- `--confidence`: Minimum confidence threshold for detections (default: `0.2`)

Example:

```bash
python object_tracking.py --input input.mp4 --confidence 0.3
```

## Output

- Processed video is saved to the `output/` directory as `.avi`
- Real-time detection and tracking displayed in a window
- Press `q` to quit the video display

## Classes Detected

The MobileNet-SSD model is trained on 20 object categories:

```
aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable,
dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor
```


## Notes

- The tracker only initializes on the first frame (or every `N` frames if modified).
- The MOSSE tracker is lightweight and fast, but may drift with fast motion or occlusion.