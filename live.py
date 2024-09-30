import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')

# Open the webcam (0 for default webcam)
cap = cv2.VideoCapture(0)

plt.ion()
fig, ax = plt.subplots()

exit_flag = False

def on_key(event):
    global exit_flag
    if event.key == 'q': 
        exit_flag = True
        
fig.canvas.mpl_connect('key_press_event', on_key)

while True:
    if exit_flag:
        break

    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

# Run YOLOv8 detection on the frame
    results = model(frame)

    detections = results[0].boxes

# Draw bounding boxes and labels on the frame
    for detection in detections:
       
        box = detection.xyxy[0].cpu().numpy()
        confidence = detection.conf.cpu().numpy()[0]
        class_id = int(detection.cls.cpu().numpy()[0])

        color = (0, 255, 0)  # ---Green color for boxes---
        label = f"{model.names[class_id]}: {confidence:.2f}"

    # Draw the bounding box
        cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(frame, label, (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    ax.imshow(frame_rgb)
    plt.draw()
    plt.pause(0.001)  
    ax.clear() 

# Release the video capture and close all plots
cap.release()
plt.close()
