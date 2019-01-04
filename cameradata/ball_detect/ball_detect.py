import freenect
import cv2
import numpy as np
import imutils
from cameradata.utils.perspTransform import four_point_transform
from cameradata.utils.get_pts_gui import get_points
from time import sleep

def get_video(pts):
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return four_point_transform(array, pts)


def get_depth(pts):
    array, _ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return four_point_transform(array, pts)


def get_ball_reference(pts):
    """Frame with no balls and no cue"""
    img = get_depth(pts)
    return img


def get_ball_diff(pts, Ra):
    return cv2.absdiff(get_depth(pts), Ra)


pts = get_points(1)
Ra = get_ball_reference(pts)
cv2.imshow("1", imutils.resize(Ra, height=320))
sleep(0)

while 1:
    ballFrame = get_ball_diff(pts, Ra)
    cv2.imshow("2", imutils.resize(ballFrame, height=320))


    Bw = 25
    Tb = (13/16)*Bw
    ballBin = np.zeros(ballFrame.shape, np.uint8)
    ballBin[ballFrame > Tb] = 255
    cv2.imshow("3", imutils.resize(ballBin, height=320))
    ballErode = cv2.erode(ballBin, np.ones((9, 9), np.uint8))
    cv2.imshow("4", imutils.resize(ballErode, height=320))
    _, contours, hierarchy = cv2.findContours(ballErode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    image = get_depth(pts)
    n = 0
    for cnt in contours:
        if cv2.contourArea(cnt) > 110:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)
            image = cv2.circle(image, center, radius, 255, 1)
            n += 1


    print(n)
    cv2.imshow("5", imutils.resize(image, height=320))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()