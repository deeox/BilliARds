import freenect
import pickle
from operator import itemgetter
from time import sleep

import cv2
import imutils
import numpy as np

from cameradata.utils.perspTransform import four_point_transform


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


def get_ball_diff(curr_depth, Ra):
    return cv2.absdiff(curr_depth, Ra)


def get_ball_contours(curr_depth, Ra):
    ballFrame = get_ball_diff(curr_depth, Ra)
    # cv2.imshow("2", imutils.resize(ballFrame, height=320))

    Bw = 26
    Tb = (13 / 16) * Bw
    ballBin = np.zeros(ballFrame.shape, np.uint8)
    ballBin[ballFrame > Tb] = 255


    # ballSmooth = cv2.medianBlur(ballBin, 5)
    ballErode = cv2.erode(ballBin, np.ones((7, 7), np.uint8))
    # cv2.imshow("4", imutils.resize(ballErode, height=320))

    contours, hierarchy = cv2.findContours(ballErode, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours


def draw_contour_circles(img, contours):
    n = 0
    centers_all = []
    radii_all = []
    for i in range(len(contours)):
        if cv2.contourArea(contours[i]) > 110:
            (x, y), radius = cv2.minEnclosingCircle(contours[i])
            center = (int(x), int(y))
            centers_all.append(center)
            radius = int(radius)
            radii_all.append(radius)
            img = cv2.drawContours(img, contours, i, 255, 1)
            img = cv2.circle(img, center, radius + 3, 255, 1)
            img = cv2.rectangle(img, (int(x - 1), int(y - 1)), (int(x + 1), int(y + 1)), 255)
            n += 1
    return img, centers_all, radii_all


def get_white_ball(img_rgb, img_depth, contours):
    global wht_center, wht_radius
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
    # print(sum_val)
    wht_center = (0, 0)
    wht_radius = 0
    max_cnt_index = None
    max_cnt = None
    if len(sum_val) > 0:
        max_cnt_index = max(sum_val, key=itemgetter(1))[0]
        cv2.drawContours(img_rgb, contours, max_cnt_index, color=(255, 120, 0), thickness=-1)
        max_cnt = contours[max_cnt_index]

    # print(max_cnt)
    if max_cnt is not None and cv2.contourArea(max_cnt) > 110:
        (x, y), radius = cv2.minEnclosingCircle(max_cnt)
        wht_center = (int(x), int(y))
        wht_radius = int(radius)

    return max_cnt_index, max_cnt, wht_center, wht_radius, img_rgb


def ball_det(curr_frame_depth, curr_frame_rgb, Ra):
    global cent_depth, cent_rgb

    cent_depth, cent_rgb = [], []
    rad_depth, rad_all = [], []
    contours = get_ball_contours(curr_frame_depth, Ra)

    img_depth = curr_frame_depth
    img_rgb = curr_frame_rgb

    img_rgb = cv2.resize(img_rgb, (img_depth.shape[1], img_depth.shape[0]))
    # image1 = cv2.drawContours(image1, contours, -1, 255, 2)

    img_depth, cent_all, rad_all = draw_contour_circles(img_depth, contours)
    img_rgb, cent_rgb_all, cent_rgb_all = draw_contour_circles(img_rgb, contours)

    #cv2.imshow("5", imutils.resize(img_depth, height=320))
    #cv2.imshow("6", imutils.resize(img_rgb, height=320))
    # noinspection PyTypeChecker
    max_cnt_idx, max_cnt, wht_cent, wht_rad, img_rgb = get_white_ball(img_rgb, img_depth=img_depth, contours=contours)

    cent_depth.append(wht_cent)
    rad_depth.append(wht_rad)
    for cent in cent_all:
        if cent != cent_depth[0]:
            cent_depth.append(cent)
    for rad in rad_all:
        if rad != rad_depth[0]:
            rad_depth.append(rad)

    return cent_depth, rad_depth

if __name__ == "__main__":
    with open('../../sys_setup/pts_depth.pkl', 'rb') as input1:
        pts_depth = pickle.load(input1)

    with open('../../sys_setup/pts_rgb.pkl', 'rb') as input2:
        pts_rgb = pickle.load(input2)

    x = get_depth(pts_depth)
    Ra = cv2.imread('../../sys_setup/ref_depth_no_balls.png')
    Ra = Ra[:, :, 0]
    Ra_rgb = cv2.imread('../../sys_setup/ref_rgb_no_balls.png')

    while 1:

        curr_frame_depth, curr_frame_rgb = get_depth(pts_depth), get_video(pts_rgb)
        img_d = curr_frame_depth.copy()
        cent_depth, rad_depth = ball_det(curr_frame_depth, curr_frame_rgb, Ra)
        print(len(cent_depth))
        print(cent_depth[0])

        for cent in cent_depth:
            if cent != cent_depth[0]:
                for rad in rad_depth:
                    cv2.circle(img_d, cent, rad +3, 0, 2)
        cv2.circle(img_d, cent_depth[0], rad_depth[0] + 3, 0, -1)
        cv2.imshow("ball_det Demo", imutils.resize(img_d, height=250))
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()
