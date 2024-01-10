import cv2

#TODO: IMPLEMENT YOLO TO CLASSIFY VEHICLES WITHIN BOUNDING BOX

def detect_cars(reference, current, parking_spaces, color_changed):
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    threshold = 50
    changed_indices = set()

    for index, parking_space in enumerate(parking_spaces):
        x1, y1, x2, y2 = parking_space

        reference_roi = gray_reference[y1:y2, x1:x2]
        current_roi = gray_current[y1:y2, x1:x2]

        diff_roi = cv2.absdiff(reference_roi, current_roi)
        _, threshold_diff_roi = cv2.threshold(diff_roi, threshold, 255, cv2.THRESH_BINARY)

        if cv2.countNonZero(threshold_diff_roi) > 0:
            changed_indices.add(index)

    unchanged_indices = set(range(len(parking_spaces))) - changed_indices
    available_spots = len(unchanged_indices)

    for index, parking_space in enumerate(parking_spaces):
        x1, y1, x2, y2 = parking_space

        if index in changed_indices:
            cv2.rectangle(current, (x1, y1), (x2, y2), (0, 0, 255), 2)
            color_changed.add(index)
        else:
            cv2.rectangle(current, (x1, y1), (x2, y2), (0, 255, 0), 2)
            if index in color_changed:
                color_changed.remove(index)

    return color_changed, available_spots
if __name__ == "__main__":
    # Main part of the code
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

    output_filename = input("Enter the filename of the output video file (without extension): ")
    out = cv2.VideoWriter(f"output/{output_filename}.avi", cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width,frame_height))

    prev_frame = None
    color_changed = set()  # Store indices of parking spaces with changed color
    available_spots = len(parking_spaces)  # Initialize available spots with total parking spaces

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if prev_frame is None:
            prev_frame = frame.copy()

        color_changed, available_spots = detect_cars(reference_image, frame, parking_spaces, color_changed)

        cv2.putText(frame, f"Available Spots: {available_spots}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Detected Cars', frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()