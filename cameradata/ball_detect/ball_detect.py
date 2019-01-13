import freenect
import cv2
import numpy as np
import imutils
from cameradata.utils.perspTransform import four_point_transform
from cameradata.utils.get_pts_gui import get_points
from time import sleep
from operator import itemgetter


def get_video(pts):
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return four_point_transform(array, pts)


def get_depth(pts):
    array, _ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return four_point_transform(array, pts)


def get_ball_reference(pts_depth, pts_rgb):
    """Frame with no balls and no cue"""
    img_depth = get_depth(pts_depth)
    img = get_video(pts_rgb)
    return img_depth, img


def get_ball_diff(pts, Ra):
    return cv2.absdiff(get_depth(pts), Ra)


def get_ball_contours(pts, Ra):
    ballFrame = get_ball_diff(pts, Ra)
    # cv2.imshow("2", imutils.resize(ballFrame, height=320))

    Bw = 29
    Tb = (13 / 16) * Bw
    ballBin = np.zeros(ballFrame.shape, np.uint8)
    ballBin[ballFrame > Tb] = 255
    # cv2.imshow("3", imutils.resize(ballBin, height=320))

    # ballSmooth = cv2.medianBlur(ballBin, 5)
    ballErode = cv2.erode(ballBin, np.ones((7, 7), np.uint8))
    # cv2.imshow("4", imutils.resize(ballErode, height=320))

    _, contours, hierarchy = cv2.findContours(ballErode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contour_circles(img, contours):
    n = 0
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 110:
            (x, y), radius = cv2.minEnclosingCircle(contours[i])
            center = (int(x), int(y))
            radius = int(radius)
            img = cv2.drawContours(img, contours, i, 255, 1)
            img = cv2.circle(img, center, radius + 3, 255, 1)
            img = cv2.rectangle(img, (int(x - 1), int(y - 1)), (int(x + 1), int(y + 1)), 255)
            n += 1
    return img


def get_white_ball(img_rgb, img_depth, contours):
    img_rgb = cv2.resize(img_rgb, (img_depth.shape[1], img_depth.shape[0]))
    # Initialize empty list
    # global max_cnt_index, max_cnt
    lst_intensities = []

    # For each list of contour points...

    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 110:
            # Create a mask image that contains the contour filled in
            cimg = np.zeros_like(img_rgb)
            cimg = cv2.resize(cimg, (img_depth.shape[1], img_depth.shape[0]))
            cv2.drawContours(cimg, contours, i, color=255, thickness=-1)

            # Access the image pixels and create a 1D numpy array then add to list
            pts = np.where(cimg == 255)
            lst_intensities.append((i, img_rgb[pts[0], pts[1]]))

    # print(lst_intensities)

    sum_val = []
    Sum = 0
    for i, cnt in lst_intensities:
        for val in cnt:
            # print(val)
            Sum += sum(val)
        sum_val.append((i, Sum))
        Sum = 0
    #print(sum_val)

    max_cnt_index = None
    max_cnt = None
    if len(sum_val) > 0:
        max_cnt_index = max(sum_val, key=itemgetter(1))[0]
        cv2.drawContours(img_rgb, contours, max_cnt_index, color=(255, 120, 0), thickness=-1)
        max_cnt = contours[max_cnt_index]

    return max_cnt_index, max_cnt, img_rgb


pts_depth = get_points(1)
pts_rgb = get_points(0)
Ra, Ra_rgb = get_ball_reference(pts_depth, pts_rgb)
# cv2.imshow("1", imutils.resize(Ra, height=320))
sleep(0)

while 1:
    contours = get_ball_contours(pts_depth, Ra)

    img_depth = get_depth(pts_depth)
    img_rgb = get_video(pts_rgb)

    img_rgb = cv2.resize(img_rgb, (img_depth.shape[1], img_depth.shape[0]))
    # image1 = cv2.drawContours(image1, contours, -1, 255, 2)

    img_depth = draw_contour_circles(img_depth, contours)
    img_rgb = draw_contour_circles(img_rgb, contours)

    cv2.imshow("5", imutils.resize(img_depth, height=320))

    max_cnt_idx, max_cnt, img_rgb = get_white_ball(img_rgb, img_depth, contours)

    cv2.imshow("6", imutils.resize(img_rgb, height=320))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
