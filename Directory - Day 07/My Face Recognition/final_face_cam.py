import joblib
import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time

# Try different camera indices if necessary
camera_index = 0
cam = cv2.VideoCapture(camera_index)

# Check if the camera opened successfully
if not cam.isOpened():
    print(f"Error: Unable to open camera with index {camera_index}")
    exit()

path = "C:\\Users\\student\\Desktop\\Day2_Pratham_AI_ML\\My Face Recognition\\haarcascade_frontalface_default.xml"
# Face detector
face_detector = cv2.CascadeClassifier(path)

model_path = 'C:\\Users\\student\\Desktop\\Day2_Pratham_AI_ML\\My Face Recognition\\my_orl_face_trained_model.pkl'
try:
    face_model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

frame = True
count = 0
while frame:
    ret, im = cam.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    try:
        im_new = cv2.resize(im, (512, 512))
    except cv2.error as e:
        print("Resize error:", e)
        continue

    # Convert the color (BGR) into grayscale
    gray_im = cv2.cvtColor(im_new, cv2.COLOR_BGR2GRAY)
    # Run your classifier on the image
    faces = face_detector.detectMultiScale(gray_im, scaleFactor=1.1, minNeighbors=10)
    # Display the bounding box on all the faces
    for (dx, dy, w, h) in faces:
        cv2.rectangle(im_new, (dx, dy), (dx + w, dy + h), (0, 0, 255), 2)
        cropped_im = gray_im[dy-20:(dy+h)+40, dx:(dx+w)]
        try:
            cropped_im_resized = cv2.resize(cropped_im, (92, 112))
        except cv2.error as e:
            print("Resize error:", e)
            continue

        lb = face_model.predict(cropped_im_resized.reshape(1, -1))
        cv2.putText(im_new, 'user: ' + str(lb[0]), (dx - 5, dy - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color=(0, 0, 255), thickness=2)

    cv2.imshow('Camera Live Feed', im_new)
    # Desired button of your choice
    if cv2.waitKey(1) & 0xFF == ord('q'):
        frame = False
        break

cam.release()
cv2.destroyAllWindows()
