# Import necessary libraries
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tensorflow.keras.models import load_model
from collections import Counter, deque

# Load the pre-trained gesture recognition model
model = load_model('models/m1-2.keras')

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Process video frames dynamically
    max_num_hands=1,          # Detect only one hand
    min_detection_confidence=0.7  # Minimum confidence threshold for detection
)

# Set up video capture with OpenCV
cap = cv2.VideoCapture(0)  # Use the default camera (webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)  # Set frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)  # Set frame height
cap.set(cv2.CAP_PROP_FPS, 30)  # Set frame rate

# Skip initial frames to allow the camera to stabilize
for i in range(100):
    cap.read()

# Initialize a deque to store predictions for temporal smoothing
predictions = deque()
prediction_text = None  # Variable to store the prediction text for display

# Main loop for processing camera frames
while cap.isOpened():
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        print("Failed to capture image")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    result = hands.process(rgb_frame)

    # Create a black frame to visualize the hand landmarks
    h, w, _ = frame.shape
    black_frame = np.zeros((h, w), dtype=np.uint8)

    # If hand landmarks are detected, process them
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Extract and scale the landmark points
            landmark_points = []
            for landmark in hand_landmarks.landmark:
                x, y = int(landmark.x * w), int(landmark.y * h)
                landmark_points.append((x, y))
                cv2.circle(black_frame, (x, y), 10, (255), -1)  # Draw landmarks

            # Draw connections between landmarks
            for connection in mp_hands.HAND_CONNECTIONS:
                start_idx = connection[0]
                end_idx = connection[1]
                start_point = landmark_points[start_idx]
                end_point = landmark_points[end_idx]
                cv2.line(black_frame, start_point, end_point, (255), 11)  # Draw connections

            # Preprocess the black frame for model input
            input_image = cv2.resize(black_frame, (224, 224))  # Resize to 224x224
            input_image = input_image.reshape((224, 224, 1))  # Add channel dimension
            input_image = input_image / 255.0  # Normalize pixel values
            input_images = np.array([input_image])  # Create batch dimension

            # Perform prediction using the loaded model
            mpredictions = model.predict(input_images)
            mpredicted_classes = np.argmax(mpredictions, axis=1)  # Get predicted class
            predictions.append(mpredicted_classes[0])  # Add to predictions deque

            # Limit the size of the deque to the last 16 predictions
            if len(predictions) >= 16:
                predictions.popleft()

            # Determine the most common prediction from the deque
            most_common_prediction = Counter(predictions).most_common(1)[0][0]
            prediction_text = f"Prediction: {most_common_prediction}"  # Format prediction text

    # Display the prediction text on the black frame
    if prediction_text:
        cv2.putText(
            black_frame, prediction_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, 
            (255, 255, 255), 2, cv2.LINE_AA
        )

    # Show the black frame with landmarks and predictions
    cv2.imshow("Camera Output - Adjust Your Hand Position", black_frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and clean up
cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows

# Close the MediaPipe Hands solution
hands.close()
