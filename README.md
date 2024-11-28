# Hand2Num

## Project Overview

This project is a real-time hand gesture recognition system that uses computer vision and deep learning technologies to classify hand gestures from webcam input. The system leverages MediaPipe for hand landmark detection and a custom Convolutional Neural Network (CNN) for gesture classification.


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

### 2. Model Training (`Project_HGR.ipynb`)
- Prepares and preprocesses image dataset
- Builds a Convolutional Neural Network (CNN)
- Trains and validates the gesture recognition model
- Saves the best performing model

### 3. Live Classification (`live_cam_test.py`)
- Loads pre-trained model
- Processes real-time webcam input
- Performs hand gesture recognition
- Displays prediction results


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
