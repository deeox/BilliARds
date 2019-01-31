import sys, os
import freenect
import cv2
import numpy as np
import imutils
import time
from cameradata.utils.perspTransform import four_point_transform
from cameradata.utils.get_pts_gui import get_points
import queue
import pickle


def get_video(pts):
    array, _ = freenect.sync_get_video()
    array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
    return four_point_transform(array, pts)


def get_depth(pts):
    array, _ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return cv2.GaussianBlur(four_point_transform(array, pts), (5, 5), 3)


def qlist(q):
    return list(q.queue)

with open('../../sys_setup/pts_depth.pkl', 'rb') as input:
    pts = pickle.load(input)

M = pts[1][0] - pts[0][0]
print(M)
Np = 6
imageQ = queue.Queue(maxsize=Np)
wt = 0  # 0 is no motion, 1 is motion at t
wt_1 = 0  # 0 is no motion, 1 is motion at t_1
Kt = 0  # Counter
K1 = 400  # 0.7 * M
K2 = 8
print(K1)
while 1:
    #cv2.imshow("origDepth", imutils.resize(get_depth(pts), height=320))
    a = time.time()
    if imageQ.full() is False:
        imageQ.put(get_depth(pts))
        continue

    motionMat = np.zeros(get_depth(pts).shape)
    for i in range(Np - 2, -1, -1):
        # noinspection PyTypeChecker
        motionMat += cv2.absdiff(qlist(imageQ)[Np - 1], qlist(imageQ)[i])

    Tm = 0.00015 * 65535
    motionBin = np.zeros(motionMat.shape)
    motionBin[motionMat >= Tm] = 1

    Cmt, _, _, _ = cv2.sumElems(motionBin)
    print(Cmt)

    if wt == 0:
        # to detect when motion starts
        if Cmt > K1 and Kt < K2:
            Kt += 1
        elif Cmt < K1 and 0 < Kt < K2:
            Kt -= 1
            if Kt < 0:
                Kt = 0
        elif Kt == K2:
            wt = 1
            Kt = 0
    elif wt == 1:
        if Cmt < K1 and Kt < K2:
            Kt += 1
        elif Cmt > K1 and 0 < Kt < K2:
            Kt -= 1
            if Kt < 0:
                Kt = 0
        elif Kt == K2:
            if wt_1 == 1:
                wt = 0
                Kt = 0
            elif wt_1 == 0:
                wt = 1
                Kt = 0
        elif wt_1 == 0 and Kt < K2:
            wt = 0
            Kt = 0
        elif wt_1 == 1 and Kt < K2:
            wt = 1
            Kt = 0

    imageQ.get()
    imageQ.put(get_depth(pts))

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(motionBin, 'wt=' + str(wt), (int(0), int(25)), font, 1, (255), 2, cv2.LINE_AA)
    cv2.putText(motionBin, 'wt_1=' + str(wt_1), (int(0), int(52)), font, 1, (255), 2, cv2.LINE_AA)
    cv2.putText(motionBin, 'Kt=' + str(Kt), (int(0), int(79)), font, 1, (255), 2, cv2.LINE_AA)
    #cv2.imshow("motionDepth", imutils.resize(motionMat, height=320))
    cv2.imshow("Binary", imutils.resize(motionBin, height=320))
    b = time.time()
    # print(1/(b-a))         #show fps
    wt_1 = wt
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break
cv2.destroyAllWindows()
