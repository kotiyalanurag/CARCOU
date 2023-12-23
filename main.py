import cv2
import cvzone
import math

from sort import *
from ultralytics import YOLO

# initializing YOLO-model nano version
model = YOLO('yolov8n.pt')

YOLO_CLASS = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# object class to be detected
CLASSES = ['car', 'bus']

# initializing SORT object for tracking detected cars
tracker = Sort(max_age = 40, min_hits = 3, iou_threshold = 0.3)

# BGR value of colors 
BGR_RED = (0, 0, 255)
BGR_GREEN = (0, 255, 0)
BGR_OLIVE = (0, 128, 0)
BGR_PEACH = (129, 129, 246)

# defining (x1, y1) and (x2, y2) coordinates for "red" line
LIMITS = [915, 1575, 2800, 1575]

# list for appending ids of cars that cross the "red" line
COUNT = []

# reading video
cap = cv2.VideoCapture('video02.mp4')

# reading frame mask (2160, 3840)
mask = cv2.imread('mask.png')

while True:
    
    ret, frame = cap.read()
    
    # defining ROI for detection using mask and image
    frameRegion = cv2.bitwise_and(frame, mask)
    
    results = model(frameRegion, stream = True)
    
    detections = np.empty((0, 5))
    
    for result in results:
        boxes = result.boxes

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # calculating width and height of bounding box
            w, h = (x2 - x1), (y2 - y1)
            
            # calculating confidence of detection and getting detection class
            confidence = math.ceil((box.conf[0] * 100)) / 100
            object_class = int(box.cls[0])
            
            detected_class = YOLO_CLASS[object_class]
            
            # detect only those "cars" that have a confidence value > 30%
            if detected_class in CLASSES and confidence > 0.3:
                currentdetection = np.array([x1, y1, x2, y2, confidence])
                detections = np.vstack((detections, currentdetection))
    
    # updating detection results for tracker
    resultsTracker = tracker.update(detections)
    
    # drawing a "red" line on frame
    cv2.line(frame, (LIMITS[0], LIMITS[1]), (LIMITS[2], LIMITS[3]), BGR_RED, 5)
    
    for result in resultsTracker:
        x1, y1, x2, y2, id = result
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = (x2 - x1), (y2 - y1)

        # drawing custom bounding boxes around detected cars
        cvzone.cornerRect(frame, (x1, y1, w, h), colorR = BGR_PEACH, 
                              colorC = BGR_OLIVE, rt = 3)

        # putting text above detected cars that displays tracking id      
        cvzone.putTextRect(frame, f'{int(id)}', (max(0, x1), max(35, y1)),
                                   colorR = BGR_PEACH, colorB = BGR_OLIVE,
                                   scale = 2, thickness = 3)

        # calcuting center coordinates of a bounding box
        cx, cy = x1+w//2, y1+h//2
        
        # checking if the center of a bounding box lies at a point on our "red" line
        if LIMITS[0] < cx < LIMITS[2] and LIMITS[1] - 30 < cy < LIMITS[1] + 30:
            if COUNT.count(id) == 0:
                COUNT.append(id)
                # changing line color to "green" in case a detected car crosses the line
                cv2.line(frame, (LIMITS[0], LIMITS[1]), (LIMITS[2], LIMITS[3]), BGR_GREEN, 5)
    
    # text for the count variable
    cvzone.putTextRect(frame, f'Count: {len(COUNT)}', (60, 150),
                                   colorR = BGR_PEACH, colorB = BGR_OLIVE, 
                                   scale = 10, thickness = 7, offset = 20)
    
    cv2.imshow('Car Counter', frame)
    cv2.waitKey(1)