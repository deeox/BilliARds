import freenect
import cv2
import numpy as np
import imutils
import time
from perspTransform import four_point_transform


def get_img():
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return array


def get_depth():
    array, _ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array


def cartesian_coord(line):
    rho, theta = line
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 3000 * (-b))
    y1 = int(y0 + 3000 * (a))
    x2 = int(x0 - 3000 * (-b))
    y2 = int(y0 - 3000 * (a))
    return x1, y1, x2, y2


def find_intersection(line1, line2):
    # extract points
    x1, y1, x2, y2 = cartesian_coord(line1[0])
    x3, y3, x4, y4 = cartesian_coord(line2[0])
    # compute determinant
    Px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / \
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    Py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / \
         ((x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return Px, Py


def segment_lines(lines, delta):
    h_lines = []
    v_lines = []
    for line in lines:
        for rho, theta in line:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 3000 * (-b))
            y1 = int(y0 + 3000 * (a))
            x2 = int(x0 - 3000 * (-b))
            y2 = int(y0 - 3000 * (a))
            if abs(x2 - x1) < delta:  # x-values are near; line is vertical
                v_lines.append(line)
            elif abs(y2 - y1) < delta:  # y-values are near; line is horizontal
                h_lines.append(line)
    return h_lines, v_lines


def cluster_points(points, nclusters):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, _, centers = cv2.kmeans(points, nclusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
    return centers


def find_req_lines(im, h_lines, v_lines, x_c, y_c):
    min_pos_x = 999999
    max_neg_x = -999999
    min_pos_y = 999999
    max_neg_y = -999999
    min_line_x = None
    max_line_x = None
    min_line_y = None
    max_line_y = None

    for line in h_lines:
        for rho, theta in line:
            b = np.sin(theta)
            y0 = b * rho
            diff = y_c - y0
            if diff > 0 and diff < min_pos_y:
                min_line_y = line
            if diff < 0 and diff > max_neg_y:
                max_line_y = line
    for line in v_lines:
        for rho, theta in line:
            a = np.cos(theta)
            x0 = a * rho
            diff = x_c - x0
            if diff > 0 and diff < min_pos_x:
                min_line_x = line  # sahi hai
            if diff < 0 and diff > max_neg_x:
                max_line_x = line

    req_h = [min_line_y, max_line_y]
    #print(req_h)
    req_v = [min_line_x, max_line_x]
    #print(req_v)
    intersectsimg = im.copy()

    Px = []
    Py = []
    for h_line in req_h:
        for v_line in req_v:
            px, py = find_intersection(h_line, v_line)
            Px.append(px)
            Py.append(py)

    for cx, cy in zip(Px, Py):
        cx = np.round(cx).astype(int)
        cy = np.round(cy).astype(int)
        cv2.circle(intersectsimg, (cx, cy), radius=10, color=[194, 145, 189], thickness=-1)  # -1: filled circle
    cv2.circle(intersectsimg, (x_c, y_c), radius=10, color=[194, 145, 189], thickness=-1)

    x0, y0, x1, y1 = cartesian_coord(min_line_y[0])
    x2, y2, x3, y3 = cartesian_coord(min_line_x[0])
    x4, y4, x5, y5 = cartesian_coord(max_line_y[0])
    x6, y6, x7, y7 = cartesian_coord(max_line_x[0])

    cv2.line(intersectsimg, (x0, y0), (x1, y1), color=[255, 0, 255], thickness=2)
    cv2.line(intersectsimg, (x2, y2), (x3, y3), color=[0, 255, 0], thickness=2)
    cv2.line(intersectsimg, (x4, y4), (x5, y5), color=[255, 0, 255], thickness=2)
    cv2.line(intersectsimg, (x6, y6), (x7, y7), color=[0, 255, 0], thickness=2)

    return req_h, req_v, zip(Px, Py), intersectsimg


def read_data(count):
    img = []
    for i in range(0, count):
        img.append(get_img())
        time.sleep(0.1)

    return img


def preprocess(images):
    processed = []
    for img in images:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_green = np.array([52, 79, 36])
        upper_green = np.array([100, 246, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        res = cv2.bitwise_and(img, img, mask=mask)
        #dst = cv2.fastNlMeansDenoisingColored(res, None, 8, 8, 7, 21)
        kernel = np.ones((6, 6), np.uint8)
        blur = cv2.GaussianBlur(res, (5, 5), 0)
        smooth = cv2.addWeighted(blur, 1.5, res, -0.5, 0)
        edges = cv2.Canny(smooth, 40, 100, apertureSize=3)
        close = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        processed.append(close)

    return processed


def HoughLines(procImgs):
    lines = []
    for img in procImgs:
        lines.extend(cv2.HoughLines(img, 1, np.pi / 180, 60))

    return np.array(lines)


def drawHoughLines(img, h_lines, v_lines):
    houghimg = img.copy()
    n = 0
    for line in h_lines:
        for rho, theta in line:
            color = [0, 0, 255]  # color hoz lines red
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 3000 * (-b))
            y1 = int(y0 + 3000 * (a))
            x2 = int(x0 - 3000 * (-b))
            y2 = int(y0 - 3000 * (a))
            cv2.line(houghimg, (x1, y1), (x2, y2), color=color, thickness=1)
            n += 1
        #if n == 10:
            #break
    #print(n)
    for line in v_lines:
        for rho, theta in line:
            color = [255, 0, 0]  # color vert lines blue
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 3000 * (-b))
            y1 = int(y0 + 3000 * (a))
            x2 = int(x0 - 3000 * (-b))
            y2 = int(y0 - 3000 * (a))
            cv2.line(houghimg, (x1, y1), (x2, y2), color=color, thickness=1)
            n += 1
        #if n == 10:
            #break
    #print(n)
    return houghimg


if __name__ == "__main__":
    a = time.time()
    images = read_data(3)

    x_c, y_c = (int(images[0].shape[1] / 2), int(images[0].shape[0] / 2))
    imgs_copy = images

    processedImgs = preprocess(images)

    lines = HoughLines(processedImgs)
    # print(lines)
    delta = 100
    h_lines, v_lines = segment_lines(lines, delta)

    cv2.imshow("Segmented Hough Lines", imutils.resize(drawHoughLines(imgs_copy[0], h_lines, v_lines), height=320))

    _, _, req_corners, req_lines = find_req_lines(imgs_copy[0], h_lines, v_lines, x_c, y_c)
    cv2.imshow("req intersections", imutils.resize(req_lines, height=320))

    corners = np.array(list(req_corners), dtype="float32")
    print(corners.shape)

    imgFinal = get_img()
    warped = four_point_transform(imgFinal, corners)
    cv2.imshow("warped", imutils.resize(warped, height=250))

    depthImg = get_depth()
    Np = 1
    #for i in range(Np):
    #    depthImg = depthImg + cv2.cvtColor(get_img(), cv2.COLOR_BGR2GRAY)
    #    print(depthImg[35, 35])


    cv2.imshow("origDepth", imutils.resize(depthImg, height=320))
    warped = four_point_transform(depthImg, corners)
    cv2.imshow("warpedDepth", imutils.resize(warped, height=250))
    b = time.time()

    print(b - a)
    cv2.waitKey()
    cv2.destroyAllWindows()


