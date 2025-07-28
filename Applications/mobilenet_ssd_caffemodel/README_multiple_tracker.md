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


