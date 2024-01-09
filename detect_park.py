import cv2
import numpy as np

#TODO: FIX BOUNDING BOX COLOR CHANGE
#TODO: REMOVE DRAWING OF BOXES INSIDE BOUNDING BOX
#TODO: ADD COUNTER FOR FREE PARKING SPACES

def detect_cars(parking_spaces, reference, current):
    # Convert the reference and current frames to grayscale
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    # Define a threshold for detecting changes
    threshold = 50

    # Create an empty mask to accumulate changes within parking space bounding boxes
    mask = np.zeros_like(gray_current)

    for parking_space in parking_spaces:
        x1, y1, x2, y2 = parking_space[0], parking_space[1], parking_space[2], parking_space[3]

        # Get the region of interest (ROI) within the parking space in the reference and current frames
        reference_roi = gray_reference[y1:y2, x1:x2]
        current_roi = gray_current[y1:y2, x1:x2]

        # Calculate absolute difference between the reference ROI and current ROI
        diff_roi = cv2.absdiff(reference_roi, current_roi)

        # Apply a threshold to identify changes within the ROI
        _, threshold_diff_roi = cv2.threshold(diff_roi, threshold, 255, cv2.THRESH_BINARY)

        # Update the mask with changes detected within the parking space bounding box
        mask[y1:y2, x1:x2] = threshold_diff_roi

    # Find contours of changed areas within the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    changed_indices = set()  # Store indices of parking spaces where changes are detected
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(current, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Check if the contour overlaps with any parking space
        for i, parking_space in enumerate(parking_spaces):
            if parking_space[0] < x + w // 2 < parking_space[2] and parking_space[1] < y + h // 2 < parking_space[3]:
                changed_indices.add(i)  # Store the index of the parking space with changes detected

    return changed_indices  # Return indices of parking spaces with changes detected

if __name__ == "__main__":
    parking_filename = input("Enter the filename of the parking spaces (with extension): ")
    parking_spaces = []
    with open(parking_filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            coords = list(map(int, line.strip().split(',')))
            parking_spaces.append(coords)

    reference_filename = input("Enter the filename of the reference image (with extension): ")
    reference_image = cv2.imread(reference_filename)
    if reference_image is None:
        print("Invalid filename or file format. Please try again.")
        exit()

    video_filename = input("Enter the filename of the video file (with extension): ")
    cap = cv2.VideoCapture(video_filename)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    out = cv2.VideoWriter('output/output_detection.avi', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    prev_frame = None
    color_changed = set()  # Store indices of parking spaces with changed color

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is None:
            prev_frame = frame.copy()

        changed_indices = detect_cars(parking_spaces, reference_image, frame)

        for index in changed_indices:
            if index not in color_changed:
                # Change the color to green (or any other color) for the parking space with detected changes
                x1, y1, x2, y2 = parking_spaces[index]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                color_changed.add(index)
            else:
                # Revert back to red (or original color) if changes are no longer detected
                x1, y1, x2, y2 = parking_spaces[index]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                color_changed.remove(index)

        cv2.imshow('Detected Cars', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
