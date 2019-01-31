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

wht_center = (110, 42)

def strikeDet(wht_center, d, curr_frame_depth, ref_depth_for_cue):

    #wht_center = centers[0]
    #d = self.d

    roi_curr = curr_frame_depth[int(wht_center[1] - d / 2):int(wht_center[1] + d / 2),
               int(wht_center[0] - d / 2):int(wht_center[0] + d / 2)]
    roi_ref = ref_depth_for_cue[int(wht_center[1] - d / 2):int(wht_center[1] + d / 2),
               int(wht_center[0] - d / 2):int(wht_center[0] + d / 2)]

    cv2.imshow("1", imutils.resize(roi_curr, height=320))
    cv2.imshow("2", imutils.resize(roi_ref, height=320))

    Sub = cv2.absdiff(roi_curr, roi_ref)
    Ts = 0.00019 * 65535
    Bin = np.zeros(Sub.shape, np.uint8)
    Bin[Sub > Ts] = 255

    cv2.imshow("3", imutils.resize(Bin, height=320))

    Cs, _, _, _ = cv2.sumElems(Bin)

    thres_ar = 300 * np.pi * (d ** 2) / 12

    print(Cs, thres_ar)

    if Cs > thres_ar:
        strike = 1
    else:
        strike = 0

    return strike

if __name__ == "__main__":
    with open('../../sys_setup/pts_depth.pkl', 'rb') as input:
        pts_depth = pickle.load(input)

    with open('../../sys_setup/pts_rgb.pkl', 'rb') as input2:
        pts_rgb = pickle.load(input2)

    ref_depth_for_cue = get_depth(pts_depth)
    wht_center = (129, 43)
    d = 40
    while 1:
        strike = strikeDet(wht_center, d, get_depth(pts_depth), ref_depth_for_cue)
        print(strike)
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    cv2.destroyAllWindows()