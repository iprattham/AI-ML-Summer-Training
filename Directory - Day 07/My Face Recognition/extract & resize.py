import cv2
import os

# Paths
input_folder = "C:\\Users\\student\\Desktop\\Day2_Pratham_AI_ML\\My Face Recognition\\orl_face\\samp_imgs"
output_folder = "C:\\Users\\student\\Desktop\\Day2_Pratham_AI_ML\\My Face Recognitione\\orl_face\\u42"
haarcascade_path = "C:\\Users\\student\\Desktop\\Day2_Pratham_AI_ML\\My Face Recognition\\haarcascade_frontalface_default.xml"

# Create output directory if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Load the face detector
face_detector = cv2.CascadeClassifier(haarcascade_path)

# Process each image in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".png"):
        # Read the image
        img_path = os.path.join(input_folder, filename)
        img = cv2.imread(img_path)
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        
        # Process each face found
        for (x, y, w, h) in faces:
            # Extract the face
            face = gray[y:y+h, x:x+w]
            
            # Resize to 112x92
            face_resized = cv2.resize(face, (92, 112))
            
            # Save the processed face
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, face_resized)
            break  # Process only the first detected face

print("Face extraction, resizing, grayscaling & all the required processing completed.")
