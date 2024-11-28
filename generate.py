# Import necessary libraries
import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands solution
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,  # Process video frames dynamically
    max_num_hands=1,          # Detect a maximum of one hand
    min_detection_confidence=0.7  # Minimum confidence threshold for detection
)

# Set up video capture with OpenCV
cap = cv2.VideoCapture(0)  # Use the default camera (webcam)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1000)  # Set camera frame width
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1000)  # Set camera frame height
cap.set(cv2.CAP_PROP_FPS, 30)  # Set camera frame rate

# Configuration variables
image_label = 1  # Label for images to classify
hand_type = 'r'  # 'r' for right hand, 'l' for left hand
flip_type = 'n'  # 'f' for flipped images, 'n' for normal images

# Skip the first few frames to allow the camera to adjust
for i in range(100):
    cap.read()

# Initialize counters for saving images
image_count = 0
target_count = 2500  # Target number of images to save

# Main loop for capturing and processing frames
while cap.isOpened() and image_count < target_count:
    ret, frame = cap.read()  # Capture a frame from the camera
    if not ret:
        print("Failed to capture image")
        break

    # Flip the frame horizontally for a mirror effect
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB format (required by MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hand landmarks
    result = hands.process(rgb_frame)

    # Create a black frame to draw the landmarks
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

            # Resize the processed frame to 224x224 (common input size for models)
            resized_landmarks = cv2.resize(black_frame, (224, 224))

            # Generate the output file path for saving the image
            output_path = f"output/{image_label}/pp_{hand_type}_{flip_type}_lm{image_count:05d}.png"

            # Save the processed image
            cv2.imwrite(output_path, resized_landmarks)
            print(f"Saved image {image_count + 1}/{target_count} to {output_path}")
            image_count += 1

            # Stop saving images once the target count is reached
            if image_count >= target_count:
                break

    # Display the black frame with landmarks to guide the user
    cv2.imshow("Landmarks Output - Adjust Your Hand Position", black_frame)

    # Break the loop if the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources and clean up
cap.release()  # Release the video capture object
cv2.destroyAllWindows()  # Close all OpenCV windows

# Close the MediaPipe Hands solution
hands.close()
