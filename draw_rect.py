##########################################
# draws a bounding box and the reference 
# image and outputs the coordinates
##########################################

import cv2

def draw_rectangle(event, x, y, flags, param):
    global parking_spaces
    if event == cv2.EVENT_LBUTTONDOWN:
        parking_spaces.append((x, y))
    elif event == cv2.EVENT_LBUTTONUP:
        parking_spaces.append((x, y))
        cv2.rectangle(param['reference_image'], parking_spaces[-2], parking_spaces[-1], (0, 255, 0), 2)

# TODO: refresh window to remove last rectangle display after undo
def undo_last_rectangle():
    global parking_spaces
    if len(parking_spaces) >= 2:
        del parking_spaces[-2:]
        reference_image_copy = reference_image.copy()
        for i in range(0, len(parking_spaces), 2):
            cv2.rectangle(reference_image_copy, parking_spaces[i], parking_spaces[i+1], (0, 255, 0), 2)
        cv2.imshow('Reference Image', reference_image_copy)
    else:
        reference_image_copy = reference_image.copy()
        cv2.imshow('Reference Image', reference_image_copy)

if __name__ == "__main__":
    reference_image = None
    while reference_image is None:
        reference_filename = input("Enter the filename of the reference image (with extension): ")
        reference_image = cv2.imread(reference_filename)
        if reference_image is None:
            print("Invalid filename or file format. Please try again.")

    parking_spaces = []

    cv2.namedWindow('Reference Image')
    params = {'reference_image': reference_image}
    cv2.setMouseCallback('Reference Image', draw_rectangle, param=params)

    while True:
        cv2.imshow('Reference Image', reference_image)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('u'):  # Press 'u' to undo the last drawn rectangle
            undo_last_rectangle()

    cv2.destroyAllWindows()

    output_filename = input("Enter the desired filename for the parking spaces (without extension): ")
    with open(f"parklib/{output_filename}.txt", 'w') as file:
        for i in range(0, len(parking_spaces), 2):
            x1, y1 = parking_spaces[i]
            x2, y2 = parking_spaces[i+1]
            file.write(f"{x1},{y1},{x2},{y2}\n")
