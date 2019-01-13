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


def get_cue_reference(pts):
    """Frame with balls and no cue"""
    img = get_depth(pts)
    return img


def get_cue_diff(pts, Qa):
    return cv2.absdiff(get_depth(pts), Qa)


def max_cnt_area(contours):
    maxArea = 0
    maxCnt = None

    for cnt in contours:
        if cv2.contourArea(cnt) > maxArea:
            maxArea = cv2.contourArea(cnt)
            maxCnt = cnt

    return maxArea, maxCnt


def get_avg_area(contours):
    Al, _ = max_cnt_area(contours)
    Nc = len(contours)
    avg = 0
    for cnt in contours:
        if cv2.contourArea(cnt) < Al:
            avg += cv2.contourArea(cnt)

    avg -= Al
    print(avg)
    avg /= Nc - 1

    return abs(100 * avg)


def get_req_cnts(contours):
    req_cnt = []
    for cnt in contours:
        if cv2.contourArea(cnt) < get_avg_area(contours):
            req_cnt.append(cnt)
    return req_cnt


def get_req_contours(contours):
    cnt_ar = []
    for cnt in contours:
        cnt_ar.append((cnt, cv2.contourArea(cnt)))

    sort_cnt_ar = sorted(cnt_ar, key=itemgetter(1), reverse=True)
    req_cnt = []
    for cnt, ar in sort_cnt_ar:
        if ar > 170:
            req_cnt.append(cnt)
    return req_cnt


pts = get_points(1)
Qa = get_cue_reference(pts)
# cv2.imshow("1", imutils.resize(Qa, height=320))

while 1:
    cueFrame = get_cue_diff(pts, Qa)
    # cv2.imshow("2", imutils.resize(cueFrame, height=320))

    Tc = 0.0001 * 65535
    cueBin = np.zeros(cueFrame.shape, np.uint8)
    cueBin[cueFrame > Tc] = 255
    # cv2.imshow("3", imutils.resize(cueBin, height=320))

    _, contours, hierarchy = cv2.findContours(cueBin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    cnt_image = np.zeros(cueFrame.shape, np.uint8)
    cnt_image_2 = cnt_image.copy()
    cnt_image_all = cv2.drawContours(cnt_image, contours, -1, 255, 1)

    # cv2.imshow("4", imutils.resize(cnt_image_all, height=320))

    req_cnt = get_req_contours(contours)
    cnt_image_req = cv2.drawContours(cnt_image_2, req_cnt, -1, 255, 1)
    kernel = np.ones((8, 8), np.uint8)
    closing = cv2.morphologyEx(cnt_image_req, cv2.MORPH_CLOSE, kernel)
    cv2.imshow("5", imutils.resize(closing, height=320))
    # print(len(contours), ",", len(req_cnt))

    img = get_depth(pts)
    img_final = img.copy()
    lines = cv2.HoughLines(closing, 1, 5 * np.pi / 180, 80)

    avg_x1 = 0
    avg_x2 = 0
    avg_y1 = 0
    avg_y2 = 0
    n = 0
    if lines is not None:
        for line in lines[:15]:
            for rho, theta in line:
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                x1 = int(x0 + 1000 * (-b))
                y1 = int(y0 + 1000 * (a))
                x2 = int(x0 - 1000 * (-b))
                y2 = int(y0 - 1000 * (a))
                avg_x1 += x1
                avg_x2 += x2
                avg_y1 += y1
                avg_y2 += y2
                n += 1

                cv2.line(img, (x1, y1), (x2, y2), 255, 2)

        avg_x1 = int(avg_x1 / n)
        avg_x2 = int(avg_x2 / n)
        avg_y1 = int(avg_y1 / n)
        avg_y2 = int(avg_y2 / n)

    cv2.imshow("6", imutils.resize(img, height=320))

    cv2.line(img_final, (avg_x1, avg_y1), (avg_x2, avg_y2), 255, 1)
    cv2.imshow("7", imutils.resize(img_final, height=320))

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
