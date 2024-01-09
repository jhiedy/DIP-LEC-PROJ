import cv2

def detect_cars(parking_spaces, current):
    occupied_spaces = 0

    for space in parking_spaces:
        x1, y1, x2, y2 = space[0], space[1], space[2], space[3]

        # Get the region of interest (ROI) for each parking space
        roi = current[y1:y2, x1:x2]

        # Convert the ROI to grayscale for analysis
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Threshold the ROI to detect changes
        _, threshold_roi = cv2.threshold(gray_roi, 50, 255, cv2.THRESH_BINARY)

        # Count the number of white pixels (changes) in the ROI
        white_pixel_count = cv2.countNonZero(threshold_roi)

        # If there are changes (white pixels) above a threshold, consider the space occupied
        if white_pixel_count > 10:  # Adjust the threshold according to your scenario
            occupied_spaces += 1

    return len(parking_spaces) - occupied_spaces, current

if __name__ == "__main__":
    parking_filename = input("Enter the filename of the parking spaces (with extension): ")
    parking_spaces = []
    with open(parking_filename, 'r') as file:
        lines = file.readlines()
        for line in lines:
            coords = list(map(int, line.strip().split(',')))
            parking_spaces.append(coords)
    print(parking_spaces)

    video_filename = input("Enter the filename of the video file (with extension): ")
    cap = cv2.VideoCapture(video_filename)

    available_spaces = len(parking_spaces)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        available, current_frame = detect_cars(parking_spaces, frame)
        available_spaces = available

        cv2.putText(current_frame, f"Available Spaces: {available_spaces}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow('Detected Cars', current_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
