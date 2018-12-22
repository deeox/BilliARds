import cv2
import freenect
import numpy as np
from cameradata.utils.perspTransform import four_point_transform, order_points

refPt = []
i = 0


# function to get RGB image from kinect
def get_video():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    print(array.shape)
    return array


# function to get depth image from kinect
def get_depth():
    array, _ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    # print(array.shape)
    return array


def click_and_crop(event, x, y, flags, param):
    # grab references to the global variables
    global refPt, i, image

    if event == cv2.EVENT_LBUTTONDOWN and i < 4:
        refPt.append((x, y))

        print(refPt)
        cv2.circle(image, (x, y), 4, (0, 0, 255), -1)
        if i > 0:
            cv2.line(image, refPt[i - 1], refPt[i], color=(0), thickness=2)
        if i == 3:
            cv2.line(image, refPt[3], refPt[0], color=(0), thickness=2)
        i += 1


def get_points(imgType=0):
    global refPt, image, i

    if imgType == 0:
        image = get_video()
    else:
        image = get_depth()

    clone = image.copy()
    cv2.namedWindow("image")
    cv2.setMouseCallback("image", click_and_crop)

    # keep looping until the 'q' key is pressed
    while True:
        # display the image and wait for a keypress
        cv2.imshow("image", image)
        key = cv2.waitKey(1) & 0xFF

        # if the 'r' key is pressed, reset the cropping region
        if key == ord("r"):
            image = clone.copy()
            refPt = []
            i = 0

        # if the 'c' key is pressed, break from the loop
        elif key == ord("c"):
            break

    # print(refPt)
    pts_order = order_points(np.array(refPt).reshape((4, 2)))
    # print(pts_order)
    warped = four_point_transform(image, pts_order)
    cv2.imshow("ROI", warped)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pts_order


if __name__ == "__main__":
    get_points(0)
