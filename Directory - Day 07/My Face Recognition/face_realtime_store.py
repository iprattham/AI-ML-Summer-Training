import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import time

# Initialize webcam
cam = cv2.VideoCapture(0)

# Path to Haar cascade XML file
path = "C:\\Users\\student\\Desktop\\Day2_Pratham_AI_ML\\My Face Recognition\\haarcascade_frontalface_default.xml"
# Face detector
face_detector = cv2.CascadeClassifier(path)

# Directory to save captured images
output_dir = "captured_images"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

frame = True
count = 0
max_images = 200

while frame and count < max_images:
    val, im = cam.read()
    if not val:
        print("Failed to capture image")
        break
    im_new = cv2.resize(im, (512, 512))
    # Convert the color (BGR) into grayscale
    gray_im = cv2.cvtColor(im_new, cv2.COLOR_BGR2GRAY)
    # Run your classifier on the image
    faces = face_detector.detectMultiScale(gray_im, scaleFactor=1.1, minNeighbors=10)
    
    # Display the bounding box on all the faces
    if len(faces) > 0:
        for (dx, dy, w, h) in faces:
            cv2.rectangle(im_new, (dx, dy), (dx + w, dy + h), (255, 0, 0), 2)
        # Save the captured image
        img_name = os.path.join(output_dir, f"image_{count:03d}.jpg")
        cv2.imwrite(img_name, im)
        count += 1
        print(f"Captured {count}/{max_images}")
    
    cv2.imshow('Camera Live Feed', im_new)
    # Desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        frame = False
        break

cam.release()
cv2.destroyAllWindows()