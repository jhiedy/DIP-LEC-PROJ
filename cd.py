import cv2
import numpy as np

# Load the reference image with parking spaces
reference_image = cv2.imread('input/parking0.jpg')
# Initialize an empty list to store parking space bounding boxes
parking_spaces = []

# Function to draw rectangles on the image
def draw_rectangle(event, x, y, flags, param):
    global reference_image, parking_spaces
    if event == cv2.EVENT_LBUTTONDOWN:
        parking_spaces.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        parking_spaces.append((x, y))
        cv2.rectangle(reference_image, parking_spaces[-2], parking_spaces[-1], (0, 255, 0), 2)

cv2.namedWindow('Reference Image')
cv2.setMouseCallback('Reference Image', draw_rectangle)

while True:
    cv2.imshow('Reference Image', reference_image)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()

# Function to detect changes in parking spaces
def detect_cars(reference, current):
    # Convert images to grayscale
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    # Define a threshold for detecting changes
    threshold = 50

    # Absolute difference between the reference and current image
    diff = cv2.absdiff(gray_reference, gray_current)
    _, threshold_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

    # Find contours of changed areas
    contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contours are within parking space bounding boxes
    for contour in contours:
        for parking_space in parking_spaces:
            x, y, w, h = cv2.boundingRect(contour)
            if parking_space[0] < x + w/2 < parking_space[2] and parking_space[1] < y + h/2 < parking_space[3]:
                cv2.rectangle(current, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return current

# Capture video from a camera or a video file
cap = cv2.VideoCapture('video_file.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect cars in the frame
    detected_frame = detect_cars(reference_image, frame)

    cv2.imshow('Detected Cars', detected_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
