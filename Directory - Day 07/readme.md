## Overview
This project covers a range of advanced machine learning techniques and image processing tasks. The main highlights include decision tree modeling, image feature extraction, face recognition, and model training on both user images and the brain-tumor dataset. Key tasks performed are:

- **Decision Tree on Diabetes Dataset**: Applied decision tree classification on the `diabetes.csv` dataset.
- **Image Processing & Feature Extraction**: Converted 2D image features into 1D, traversed and greyscaled images, and visualized them in a 5x5 grid of 25 random images.
- **Face Recognition System**: Built and trained a face recognition model that:
  - Fetches an image based on a user ID input.
  - Matches a user's query image with another image of the same user from a pool of images.
  - Trained the system to recognize the developer’s face and validated it using Support Vector Machine (SVM) classification.
  - Saved the final model as a pickle file for future use.
- **Brain Tumor Detection with Face Recognition**: Extended the face recognition model to perform similar tasks on the brain-tumor dataset, identifying and recognizing faces in medical images.

## Key Functionalities
1. **Decision Tree for Diabetes Dataset**:
   - Performed classification on `diabetes.csv` using decision tree.
   
2. **Image Processing**:
   - Converted 2D features of images into 1D vectors.
   - Greyscaled and traversed the images.
   - Created a 5x5 grid of subplots displaying 25 random images.

3. **Face Recognition**:
   - Enter a user ID to display a random image of the user.
   - The model fetches a matching image of the same user from a pool of images based on the query.
   - Trained a personalized face recognition system by training the model on the developer’s own images.
   - Implemented SVM to improve face recognition accuracy.
   - Saved the trained model as a pickle file for future use.

4. **Brain Tumor Dataset Analysis**:
   - Performed face recognition and classification tasks on the brain-tumor dataset.

## Requirements
- Python 3.x
- Libraries:
  - `scikit-learn`
  - `OpenCV`
  - `matplotlib`
  - `numpy`
  - `pickle`

## Usage
1. **Run the decision tree classification** on `diabetes.csv`.
2. **Process images** by converting 2D to 1D, greyscaling, and visualizing.
3. **Face Recognition**: Enter a user ID to fetch an image and match it with another from the pool.
4. **Train the model** on personalized images for face recognition.
5. **Apply SVM** to the trained model and validate its accuracy.
6. **Perform similar tasks on the brain-tumor dataset** for medical image analysis.

## Conclusion
This project demonstrates the implementation of multiple machine learning techniques, such as decision trees, SVM, and face recognition, applied to user images and medical datasets. The trained models can accurately classify and recognize images, with potential extensions for various domains like health diagnostics.
