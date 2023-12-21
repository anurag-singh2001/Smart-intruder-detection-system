# import cv2
# import numpy as np
# import threading
# import os

# # Set environment variable
# os.environ["OPENCV_FFMPEG_THREAD_SYNC"] = "0"

# # Load YOLOv3 Tiny model
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# # Load YOLOv3 classes (coco.names contains class labels)
# with open("coco.names", "r") as f:
#     classes = f.read().strip().split("\n")

# # Open a video stream (you can also use a webcam by passing 0 as the argument)
# video_stream = cv2.VideoCapture("sample_video.mp4", cv2.CAP_FFMPEG)  # Replace with your video file

# # Check if video stream opened successfully
# if not video_stream.isOpened():
#     print("Error: Could not open video stream.")
#     exit()

# # Function for person detection
# def detect_person(frame):
#     # Get image dimensions
#     height, width = frame.shape[:2]

#     # Create a blob from the frame (preprocess)
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320, 320), swapRB=True, crop=False)

#     # Set the input for YOLO network
#     net.setInput(blob)

#     # Get output layer names
#     output_layers = net.getUnconnectedOutLayersNames()

#     # Perform forward pass to get detections
#     detections = net.forward(output_layers)

#     # Loop over the detections
#     for detection in detections:
#         for obj in detection:
#             scores = obj[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.5 and classes[class_id] == "person":
#                 # Get bounding box coordinates
#                 center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype(int)

#                 # Calculate the top-left corner of the bounding box
#                 x, y = int(center_x - w / 2), int(center_y - h / 2)

#                 # Draw the bounding box and label
#                 color = (0, 255, 0)  # Green
#                 cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#                 cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Display the result
#     cv2.imshow("Person Detection", frame)

# # Thread for person detection
# def detection_thread():
#     while True:
#         ret, frame = video_stream.read()

#         if not ret:
#             print("Error: Failed to capture frame.")
#             break

#         # Call the person detection function
#         detect_person(frame)

#         # Wait for a key press
#         if cv2.waitKey(30) & 0xFF == ord('q'):
#             break

# # Start the detection thread
# thread = threading.Thread(target=detection_thread)
# thread.start()

# # Continue capturing frames in the main thread
# while True:
#     # Wait for a key press
#     key = cv2.waitKey(30) & 0xFF
#     if key == ord('q'):
#         break

# # Release the video stream and close all windows
# video_stream.release()
# cv2.destroyAllWindows()


# import cv2
# import numpy as np
# import os

# # Set environment variable
# os.environ["OPENCV_FFMPEG_THREAD_SYNC"] = "0"

# # Load YOLOv3 Tiny model
# net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# # Load YOLOv3 classes (coco.names contains class labels)
# with open("coco.names", "r") as f:
#     classes = f.read().strip().split("\n")

# # Open a video stream (you can also use a webcam by passing 0 as the argument)
# video_stream = cv2.VideoCapture("sample_video.mp4", cv2.CAP_FFMPEG)  # Replace with your video file

# # Check if video stream opened successfully
# if not video_stream.isOpened():
#     print("Error: Could not open video stream.")
#     exit()

# # Function for person detection
# def detect_person(frame):
#     # Get image dimensions
#     height, width = frame.shape[:2]

#     # Create a blob from the frame (preprocess)
#     blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

#     # Set the input for YOLO network
#     net.setInput(blob)

#     # Get output layer names
#     output_layers = net.getUnconnectedOutLayersNames()

#     # Perform forward pass to get detections
#     detections = net.forward(output_layers)

#     # Lists to store bounding boxes and confidences
#     bboxes = []
#     confidences = []

#     # Loop over the detections
#     for detection in detections:
#         for obj in detection:
#             scores = obj[5:]
#             class_id = np.argmax(scores)
#             confidence = scores[class_id]

#             if confidence > 0.5 and classes[class_id] == "person":
#                 # Get bounding box coordinates
#                 center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype(int)

#                 # Append bounding box and confidence to lists
#                 bboxes.append([center_x, center_y, w, h])
#                 confidences.append(float(confidence))

#     # Apply Non-Maximum Suppression if there are detections
#     indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.4)

#     # Check if indices is not None and not empty
#     if indices is not None and len(indices) > 0:
#         for i in indices.flatten():
#             box = bboxes[i]
#             x, y, w, h = box[0], box[1], box[2], box[3]

#             # Draw the bounding box and label
#             color = (0, 255, 0)  # Green
#             cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
#             cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#     # Display the result
#     cv2.imshow("Person Detection", frame)

# # Continue capturing frames and performing person detection
# while True:
#     ret, frame = video_stream.read()

#     if not ret:
#         print("Error: Failed to capture frame.")
#         break

#     # Call the person detection function
#     detect_person(frame)

#     # Wait for a key press
#     if cv2.waitKey(30) & 0xFF == ord('q'):
#         break

# # Release the video stream and close all windows
# video_stream.release()
# cv2.destroyAllWindows()


import cv2
import numpy as np
import os

# Set environment variable
os.environ["OPENCV_FFMPEG_THREAD_SYNC"] = "0"

# Load YOLOv3 Tiny model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load YOLOv3 classes (coco.names contains class labels)
with open("coco.names", "r") as f:
    classes = f.read().strip().split("\n")

# Open a video stream (you can also use a webcam by passing 0 as the argument)
video_stream = cv2.VideoCapture("sample_video.mp4", cv2.CAP_FFMPEG)  # Replace with your video file

# Check if video stream opened successfully
if not video_stream.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Function for person detection
def detect_person(frame):
    # Resize frame to a smaller resolution for faster processing
    frame = cv2.resize(frame, (640, 480))

    # Get image dimensions
    height, width = frame.shape[:2]

    # Create a blob from the frame (preprocess)
    blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)

    # Set the input for YOLO network
    net.setInput(blob)

    # Get output layer names
    output_layers = net.getUnconnectedOutLayersNames()

    # Perform forward pass to get detections
    detections = net.forward(output_layers)

    # Lists to store bounding boxes and confidences
    bboxes = []
    confidences = []

    # Loop over the detections
    for detection in detections:
        for obj in detection:
            scores = obj[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5 and classes[class_id] == "person":
                # Get bounding box coordinates
                center_x, center_y, w, h = (obj[0:4] * np.array([width, height, width, height])).astype(int)

                # Append bounding box and confidence to lists
                bboxes.append([center_x, center_y, w, h])
                confidences.append(float(confidence))

    # Apply Non-Maximum Suppression if there are detections
    indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.4)

    # Check if indices is not None and not empty
    if indices is not None and len(indices) > 0:
        for i in indices.flatten():
            box = bboxes[i]
            x, y, w, h = box[0], box[1], box[2], box[3]

            # Draw the bounding box and label
            color = (0, 255, 0)  # Green
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "Person", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Display the result
    cv2.imshow("Person Detection", frame)

# Continue capturing frames and performing person detection
while True:
    ret, frame = video_stream.read()

    if not ret:
        print("Error: Failed to capture frame.")
        break

    # Call the person detection function
    detect_person(frame)

    # Wait for a key press
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the video stream and close all windows
video_stream.release()
cv2.destroyAllWindows()
