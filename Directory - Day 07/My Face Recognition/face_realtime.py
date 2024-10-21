import cv2 
import numpy as np
import matplotlib.pyplot as plt
import time

cam = cv2.VideoCapture(0)
path = "C:\\Users\\student\\Desktop\\Day2_Pratham_AI_ML\\My Face Recognition\\haarcascade_frontalface_default.xml"
# face detector 
face_detector = cv2.CascadeClassifier(path)

frame=True
while(frame==True):
    val,im = cam.read()
    im_new = cv2.resize(im, (512,512))
    # covert the color (BGR) into grayscale
    gray_im = cv2.cvtColor(im_new,cv2.COLOR_BGR2GRAY)
    # run your classifier on the image
    faces = face_detector.detectMultiScale(gray_im,scaleFactor=1.1,minNeighbors=10)
    
    # disply the bounding box on all the faces
    for var in range(len(faces)):
        dx = faces[var][0]
        dy = faces[var][1]
        w = faces[var][2]
        h = faces[var][3]
        cv2.rectangle(im_new, (dx,dy),(dx+w,dy+h),(255,0,0),2)
    cv2.imshow('Camera Live Feed', im_new)
    # desired button of your choice 
    if cv2.waitKey(1) & 0xFF == ord('q'):
        frame=False
        break
  

cam.release()
cv2.destroyAllWindows()