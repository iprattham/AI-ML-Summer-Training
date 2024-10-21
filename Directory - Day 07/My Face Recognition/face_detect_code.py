# !pip install opencv-python

# read the image 
import cv2
import numpy as np 
import matplotlib.pyplot as plt

path = 'C:\\Users\\student\\Desktop\\Day2_Pratham_AI_ML\\My Face Recognition\\image_faces.png'
im = cv2.imread(path)
im_new = cv2.resize(im, (512,512))
print("Resolution of the image")
print(im.shape)
#cv2.imshow("Multi faces image", im)

# covert the color (BGR) into grayscale
gray_im = cv2.cvtColor(im_new,cv2.COLOR_BGR2GRAY)
print("Resolution of the gray image")
print(gray_im.shape)
#cv2.imshow("Multi faces image", gray_im)


path = "C:\\Users\\student\\Desktop\\Day2_Pratham_AI_ML\\My Face Recognition\\haarcascade_frontalface_default.xml"
# face detector 
face_detector = cv2.CascadeClassifier(path)

# run your classifier on the image
faces = face_detector.detectMultiScale(gray_im,scaleFactor=1.1,minNeighbors=10)
print(faces)
all_faces=[]
# diaply the bounding box on all the faces
for var in range(len(faces)):
    dx = faces[var][0]
    dy = faces[var][1]
    w = faces[var][2]
    h = faces[var][3]
    cv2.rectangle(im_new, (dx,dy),(dx+w,dy+h),(255,0,0),2)
    # seperate out the faces 
    croppedFace = gray_im[dy:dy+h,dx:dx+w]
    all_faces.append([croppedFace])
    print(croppedFace.shape)
print(len(all_faces))

for i in range(len(all_faces)):
    f = np.array(all_faces[i])[0,:,:]
    newF = cv2.resize(f, (112,92))
    print(newF.shape)
    plt.figure(i+1)
    plt.imshow(f,cmap='gray')
    plt.title('Detected face')
    plt.axis('off')

#cv2.imshow("face detected",im_new)
#cv2.waitKey(0)
#cv2.destroyAllWindows()