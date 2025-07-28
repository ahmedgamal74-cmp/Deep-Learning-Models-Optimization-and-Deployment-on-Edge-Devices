import numpy as np
import argparse
import imutils
import cv2
import time
import os
from imutils.video import FPS

# Ensure output directory exists
if not os.path.isdir("output"):
    os.mkdir("output")

# Define class labels for MobileNetSSD
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", 
            "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
            "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
            "sofa", "train", "tvmonitor"]

# Load MobileNetSSD model
proto = 'mobilenet_ssd/MobileNetSSD_deploy.prototxt'
model = 'mobilenet_ssd/MobileNetSSD_deploy.caffemodel'

# Argument parser
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=False, help="Path to input video file (leave empty for webcam)")
ap.add_argument("-c", "--confidence", type=float, default=0.2, help="Minimum probability to filter weak detections")
args = vars(ap.parse_args())

print("[INFO] Loading model...")
net = cv2.dnn.readNetFromCaffe(proto, model)

# Video source (file or webcam)
if args["input"]:
    print("[INFO] Opening video file...")
    vs = cv2.VideoCapture(args["input"])
    output_filename = os.path.splitext(os.path.basename(args["input"]))[0]
else:
    print("[INFO] Starting webcam stream...")
    vs = cv2.VideoCapture(0)  # Webcam input
    output_filename = "webcam_output"

# Initialize tracker
multi_tracker = cv2.legacy.MultiTracker_create()
fps = FPS().start()
writer = None
labels = []

while True:
    grabbed, frame = vs.read()
    if not grabbed:
        break  # Stop if the video ends

    frame = imutils.resize(frame, width=600)

    # Initialize video writer
    if writer is None:
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        writer = cv2.VideoWriter(f"output/{output_filename}.avi", fourcc, 30, 
                                 (frame.shape[1], frame.shape[0]), True)

    # Detect objects only in the first frame
    if len(labels) == 0:
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (w, h), 127.5)
        net.setInput(blob)
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > args["confidence"]:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]

                # Get bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Initialize and add tracker
                tracker = cv2.legacy.TrackerMOSSE_create()
                multi_tracker.add(tracker, frame, (startX, startY, endX - startX, endY - startY))
                labels.append(label)

                # Draw detection box
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv2.putText(frame, label, (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)

    else:
        success, boxes = multi_tracker.update(frame)
        for i, box in enumerate(boxes):
            (startX, startY, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (startX, startY), (startX + w, startY + h), (0, 255, 0), 2)
            cv2.putText(frame, labels[i], (startX, startY - 15), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)

    # Write to output file
    if writer is not None:
        writer.write(frame)

    # Show frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break  # Quit on 'q'

    fps.update()

# Stop FPS counter and display results
fps.stop()
print("[INFO] Elapsed time: {:.2f} seconds".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

# Cleanup
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
vs.release()
