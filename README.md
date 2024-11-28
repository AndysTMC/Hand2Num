# Hand2Num

## Project Overview

This project is a real-time hand gesture recognition system that uses computer vision and deep learning technologies to classify hand gestures from webcam input. The system leverages MediaPipe for hand landmark detection and a custom Convolutional Neural Network (CNN) for gesture classification.
![Theme_Image_Transparent](https://github.com/user-attachments/assets/a34aa079-684e-4368-8e70-8a9d87b1cdcd)


## Key Features

- Real-time hand gesture recognition
- Uses MediaPipe for hand landmark detection
- Custom CNN model for gesture classification
- Developed and trained on Google Colab
- Supports multiple gesture categories

### Key Training Environment Features

- Direct Google Drive file access
- Compressed image dataset handling
- Automated model training and checkpointing
- GPU/TPU acceleration for faster computations


## Technologies Used

- **Development Platform**: Google Colab
- **Hardware Acceleration**: TPU
- **Computer Vision**: OpenCV (cv2)
- **Hand Tracking**: MediaPipe
- **Deep Learning**: TensorFlow/Keras
- **Programming Language**: Python


## Project Structure

### 1. Data Generation (`generate.py`)
- Captures hand landmark images using webcam
- Processes and saves landmark images for training
- Supports different hand configurations (left/right, normal/flipped)
#### Preprocessed Images
##### One-Left-Normal | One-Right Normal | One-Left Flipped | One-Right Flipped
<img src="https://github.com/user-attachments/assets/cfb9ef40-a4b5-436c-8aa6-9b6b50a50e81" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/3fd52b95-3d00-4b2a-82f1-70b1edd7d644" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/f1ebfa43-1f92-4649-acbc-a6210d230cb8" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/fafc9dd2-06ce-4e55-bee0-12efd94e8bea" alt="Image 1" width="200"/>

##### Two-Left-Normal | Two-Right Normal | Two-Left Flipped | Two-Right Flipped
<img src="https://github.com/user-attachments/assets/cca75418-486f-4ee2-b117-c7943655ed18" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/05156f92-fed2-4260-90bf-fe365ceeac59" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/a8594e12-847d-41a3-bc83-255158a2503f" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/8f1e0b95-b3b3-43c6-ac98-7e7ef3390d0d" alt="Image 1" width="200"/>

##### Three-Left-Normal | Three-Right Normal | Three-Left Flipped | Three-Right Flipped
<img src="https://github.com/user-attachments/assets/8e687524-cd90-4252-9a9e-f414b783c1f5" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/fffe768d-7a30-4b4e-97fb-072101831cd2" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/a095a547-ffc4-4dfd-a462-267124bd2015" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/3e137109-d1d4-4551-b312-29210f742144" alt="Image 1" width="200"/>

##### Four-Left-Normal | Four-Right Normal | Four-Left Flipped | Four-Right Flipped
<img src="https://github.com/user-attachments/assets/049f96fe-3da6-4a1e-beab-12e42d08662e" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/f577d18f-ce00-4d3f-aee9-b3e80a3f9c54" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/0590b82e-c2c3-4fdd-830b-f00534432982" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/be261bc3-39f4-4128-9795-c8b5f99153ca" alt="Image 1" width="200"/>

##### Five-Left-Normal | Five-Right Normal | Five-Left Flipped | Five-Right Flipped
<img src="https://github.com/user-attachments/assets/a7decded-e83a-4fe3-84bd-0a074326ea28" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/f7a30dbc-905b-4f69-a576-491b38de19fb" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/25cff253-09c3-4469-9192-5f6b21cec02f" alt="Image 1" width="200"/>
<img src="https://github.com/user-attachments/assets/b89d1bf3-7b02-4588-812a-f44467993050" alt="Image 1" width="200"/>


### 2. Model Training (`Project_HGR.ipynb`)
- Prepares and preprocesses image dataset
- Builds a Convolutional Neural Network (CNN)
- Trains and validates the gesture recognition model
- Saves the best performing model
### Model Architecture
<img src="https://github.com/user-attachments/assets/6f82f1ee-f4f0-4d5b-b349-76d57b817959" alt="Image 1" width="900"/>

### Few Testing Results
<img src="https://github.com/user-attachments/assets/96399dd8-9fb6-4bdd-96c8-b7bb46fc11d4" alt="Image 1" width="900"/>

### 3. Live Classification (`live_cam_test.py`)
- Loads pre-trained model
- Processes real-time webcam input
- Performs hand gesture recognition
- Displays prediction results
### Some Real-time Testing Results
<img src="https://github.com/user-attachments/assets/996dbc5b-e6cb-405f-a839-8947162bbe01" alt="Image 1" width="300"/>
<img src="https://github.com/user-attachments/assets/ade4e0e9-375f-47e3-bff5-9a9e593b02ab" alt="Image 2" width="300"/>
<img src="https://github.com/user-attachments/assets/6e78e9d6-b8ab-4855-a75c-981077c975f4" alt="Image 3" width="300"/>
<img src="https://github.com/user-attachments/assets/b6dac283-8f1a-4273-b441-d30f05bfa6be" alt="Image 4" width="300"/>
<img src="https://github.com/user-attachments/assets/7d1ebde8-cc89-4522-bde5-c216e18f9112" alt="Image 5" width="300"/>


## Setup and Reproduction

### Prerequisites
- Google Account
- Google Colab access
- Prepared image dataset

### Steps to Reproduce
1. Open Google Colab
2. Create new notebook
3. Upload or link to required Python scripts
4. Mount Google Drive
5. Upload compressed image dataset
6. Run training notebook (Project_HGR.ipynb)


## Model Deployment

After training in Colab:
- Download the best performing model
- Use `live_cam_test.py` for real-time gesture recognition
- Ensure all dependencies are installed locally

## Potential Improvements
- Increase training dataset diversity
- Implement data augmentation
- Experiment with model architectures
- Add more gesture categories

## Limitations
- Requires good lighting conditions
- Performance depends on training data quality
- Currently supports a limited number of gesture categories
