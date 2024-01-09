import cv2

#TODO: FIX TRACKING AND OVERLAPPING CLASSES
#TODO: COUNT UNIQUE CARS

if __name__ == "__main__":
    # adjust as needed
    Conf_threshold = 0.4
    NMS_threshold = 0.3

    class_name = []
    with open('yolo_setup/coco.names', 'r') as f:
        class_name = [cname.strip() for cname in f.readlines()]

    net = cv2.dnn.readNet("yolo_setup/yolov4.weights", "yolo_setup/yolov4.cfg")

    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

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

        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)

        for classid, score, box in zip(classes, scores, boxes):
            if (score > Conf_threshold) and (classid in [2, 5, 7]):
                color = (0, 255, 0)
                class_label = class_name[int(classid)]
                label = "%s : %f" % (class_label, score)
                cv2.rectangle(frame, box, color, 1)
                cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_COMPLEX, 0.3, color, 1)

        cv2.imshow("Parking Lot", frame)
        # press 'q' to terminate the window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
