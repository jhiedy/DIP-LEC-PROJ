import cv2
import numpy as np

if __name__ == "__main__":
    # Load YOLO
    net = cv2.dnn.readNet("yolo_setup/yolov4.weights", "yolo_setup/yolov4.cfg")
    classes = []
    with open("yolo_setup/coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

    # capture video
    cap = cv2.VideoCapture('input/stationary.mp4')

    # video information
    vid_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    vid_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    vid_fps = cap.get(cv2.CAP_PROP_FPS)
    print("Video size = ", vid_width, " x ", vid_height, " ")
    print("Video FPS = ", vid_fps, " ")

    car_count = 0  # counter for detected cars

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        height, width, channels = frame.shape

        # cetecting objects
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        # chowing information on the screen
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:  # TODO: FIX TRACKING
                    # object detected - process further based on class_id and coordinates
                    # If class_id corresponds to '2:car' or '5:bus' or '7:truck
                    if class_id == 2 or class_id == 5 or class_id == 7:
                        # process the bounding box coordinates (detection[:4]) of the detected object
                        # use these coordinates to determine parking space occupancy
                        x, y, w, h = detection[0:4] * np.array([width, height, width, height])
                        x, y, w, h = int(x), int(y), int(w), int(h)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        label = 'Car'  # label for cars
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        car_count += 1  # TODO: ONLY COUNT CAR ONCE

        # display result with car count
        cv2.putText(frame, f'Car Count: {car_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Parking Lot", frame)
        # press 'q' to terminate the window 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()