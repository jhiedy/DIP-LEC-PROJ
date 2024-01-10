import cv2
import numpy as np

def detect_cars(reference, current, parking_spaces, color_changed):
    gray_reference = cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)

    threshold = 50
    changed_indices = set()
    parking_status = []

    for index, parking_space in enumerate(parking_spaces):
        x1, y1, x2, y2 = parking_space

        reference_roi = gray_reference[y1:y2, x1:x2]
        current_roi = gray_current[y1:y2, x1:x2]

        diff_roi = cv2.absdiff(reference_roi, current_roi)
        _, threshold_diff_roi = cv2.threshold(diff_roi, threshold, 255, cv2.THRESH_BINARY)

        if cv2.countNonZero(threshold_diff_roi) > 0:
            changed_indices.add(index)
            parking_status.append("Occupied")
        else:
            parking_status.append("Available")

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

    return color_changed, available_spots, parking_status

if __name__ == "__main__":
    # Default configurations
    default_config = {
        "parking_filename": "parklib/demo.txt",
        "reference_filename": "input/demo-ref.jpg",
        "video_filename": "input/demo.mp4",
        "output_filename": "demo-output"
    }

    # Prompt for using default configuration
    use_default = input("Do you want to use the pre-configured options? (y/n): ").lower()
    if use_default == 'y':
        config = default_config
    else:
        config = {
            "parking_filename": input("Enter the filename of the parking spaces (with extension): "),
            "reference_filename": input("Enter the filename of the reference image (with extension): "),
            "video_filename": input("Enter the filename of the video file (with extension): "),
            "output_filename": input("Enter the filename of the output video file (without extension): ")
        }

    # Load parking spaces
    parking_spaces = []
    with open(config["parking_filename"], 'r') as file:
        lines = file.readlines()
        for line in lines:
            coords = list(map(int, line.strip().split(',')))
            parking_spaces.append(coords)

    # Load reference image
    reference_image = cv2.imread(config["reference_filename"])
    if reference_image is None:
        print("Invalid filename or file format. Please try again.")
        exit()

    # Setup video capture
    cap = cv2.VideoCapture(config["video_filename"])

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    sidebar_width = 200

    # Setup video writer
    out = cv2.VideoWriter(f"output/{config['output_filename']}.avi", cv2.VideoWriter_fourcc('M','J','P','G'), fps, (frame_width + sidebar_width, frame_height))

    color_changed = set()  # Store indices of parking spaces with changed color

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        color_changed, available_spots, parking_status = detect_cars(reference_image, frame, parking_spaces, color_changed)

        # Draw sidebar
        frame_with_sidebar = np.zeros((frame_height, frame_width + sidebar_width, 3), dtype=np.uint8)
        frame_with_sidebar[:, :frame_width] = frame  # Copy original frame

        # Display parking status
        for i, status in enumerate(parking_status):
            text = f"Spot {i}: {status}"
            color = (0, 255, 0) if status == "Available" else (0, 0, 255)
            cv2.putText(frame_with_sidebar, text, (frame_width + 10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame_with_sidebar, f"Available Spots: {available_spots}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.imshow('Parking Space Detection', frame_with_sidebar)
        out.write(frame_with_sidebar)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
