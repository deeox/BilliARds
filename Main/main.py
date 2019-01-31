import freenect
import os
import pickle
import queue
import threading
import time

import cv2
import numpy as np
import pygame

import cameradata.ball_detect.ball_detect as bd
import cameradata.cue_detect.cue_detect as cd
from cameradata.utils.perspTransform import four_point_transform

global wt, strike, centers, reflections, cue_line
n = 0


class MainApplication():
    def __init__(self, pts_depth):
        global wt, strike, centers, reflections, cue_line
        pygame.init()
        # self.fs = pygame.FULLSCREEN
        self.pts_depth = pts_depth
        self.FPS = 1000
        self.window = pygame.display.set_mode((0, 0))
        self.clock = pygame.time.Clock()
        self.window_w, self.window_h = pygame.display.get_surface().get_size()
        self.M = int(self.pts_depth[1][0] - self.pts_depth[0][0])
        self.N = int(self.pts_depth[3][1] - self.pts_depth[0][1])
        self.thickness = 5
        self.circle_countour_scale = 1.5
        self.offset = 10

        self.scalex = (self.window_w / self.M)
        self.scaley = (self.window_h / self.N)
        self.root_2_times_2 = 2.828
        self.scale = 2
        self.R = 40

    def GUI(self):
        window_w = self.window_w
        window_h = self.window_h
        window = self.window
        clock = self.clock
        FPS = self.FPS
        radius = self.R
        blue = (0, 0, 255)
        black = (0, 0, 0)
        white = (255, 255, 255)
        running = True
        root_2_times_2 = self.root_2_times_2
        offset = self.offset

        scalex = self.scalex
        scaley = self.scaley
        M = self.M
        N = self.N
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT or event.type == pygame.K_SPACE:
                    os._exit(0)

            window.fill(black)
            pygame.draw.rect(window, white, [0, 0, M * scalex, N * scaley],
                             self.thickness)
            for i in range(len(centers)):
                pygame.draw.circle(window, blue, [int(centers[i][0] * scalex), int(centers[i][1] * scaley)], radius)

            if len(reflections) > 0:
                pygame.draw.line(window, white, [int(centers[0][0] * scalex), int(centers[0][1] * scaley)],
                                 [int(reflections[0][0] * scalex), int(reflections[0][1] * scaley)], self.thickness)

            for i in range(len(reflections) - 1):
                pygame.draw.line(window, white, [int(reflections[i][0] * scalex), int(reflections[i][1] * scaley)],
                                 [int(reflections[i + 1][0] * scalex), int(reflections[i + 1][1] * scaley)],
                                 self.thickness)
                pygame.draw.circle(window, white, [int(reflections[i][0] * scalex), int(reflections[i][1] * scaley)],
                                   radius)

            pygame.display.update()
            clock.tick(FPS)
        pygame.quit()
        os._exit(0)


class MotionDetect(threading.Thread):
    def __init__(self, pts_depth):
        threading.Thread.__init__(self)

        self.pts_depth = pts_depth
        self.Np = 6
        self.imageQ = queue.Queue(maxsize=self.Np)
        self.wt = 0  # 0 is no motion, 1 is motion at t
        self.wt_1 = 0  # 0 is no motion, 1 is motion at t_1
        self.Kt = 0  # Counter
        self.K1 = 400  # 0.7 * M
        self.K2_up = 2
        self.K2_down = 8

    def get_depth(self):
        array, _ = freenect.sync_get_depth()
        array = array.astype(np.uint8)
        return cv2.GaussianBlur(four_point_transform(array, self.pts_depth), (5, 5), 3)

    def qlist(self):
        return list(self.imageQ.queue)

    def run(self):
        global wt_1, wt, isQFull

        wt = self.wt  # 0 is no motion, 1 is motion at t
        wt_1 = self.wt_1  # 0 is no motion, 1 is motion at t_1
        Kt = self.Kt  # Counter
        K1 = self.K1  # 0.7 * M

        Np = self.Np
        K2_up = self.K2_up
        K2_down = self.K2_down

        while 1:
            # cv2.imshow("origDepth", imutils.resize(get_depth(pts), height=320))
            isQFull = self.imageQ.full()
            if isQFull is False:
                self.imageQ.put(self.get_depth())
                continue

            motionMat = np.zeros(self.get_depth().shape)
            for i in range(Np - 2, -1, -1):
                # noinspection PyTypeChecker
                motionMat += cv2.absdiff(self.qlist()[Np - 1], self.qlist()[i])

            Tm = 0.00015 * 65535
            motionBin = np.zeros(motionMat.shape)
            motionBin[motionMat >= Tm] = 1

            Cmt, _, _, _ = cv2.sumElems(motionBin)
            # print(Cmt)

            if wt == 0:
                # to detect when motion starts
                if Cmt > K1 and Kt < K2_up:
                    Kt += 1
                elif Cmt < K1 and 0 < Kt < K2_up:
                    Kt -= 1
                    if Kt < 0:
                        Kt = 0
                elif Kt == K2_up:
                    wt = 1
                    Kt = 0
            elif wt == 1:
                if Cmt < K1 and Kt < K2_down:
                    Kt += 1
                elif Cmt > K1 and 0 < Kt < K2_down:
                    Kt -= 1
                    if Kt < 0:
                        Kt = 0
                elif Kt == K2_down:
                    if wt_1 == 1:
                        wt = 0
                        Kt = 0
                    elif wt_1 == 0:
                        wt = 1
                        Kt = 0
                elif wt_1 == 0 and Kt < K2_down:
                    wt = 0
                    Kt = 0
                elif wt_1 == 1 and Kt < K2_down:
                    wt = 1
                    Kt = 0

            self.imageQ.get()
            self.imageQ.put(self.get_depth())

            wt_1 = wt


class Process(threading.Thread):
    def __init__(self, pts_depth, pts_rgb, ref_depth_no_balls, ref_rgb_no_balls):
        threading.Thread.__init__(self)
        self.d = 17.1
        self.pts_depth = pts_depth
        self.pts_rgb = pts_rgb
        self.ref_depth_no_balls = ref_depth_no_balls
        self.ref_rgb_no_balls = ref_rgb_no_balls
        self.curr_frame_depth = self.get_depth()
        self.curr_frame_rgb = self.get_video()
        self.M = int(self.pts_depth[1][0] - self.pts_depth[0][0])
        self.N = int(self.pts_depth[3][1] - self.pts_depth[0][1])

    def get_video(self):
        array, _ = freenect.sync_get_video()
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        return four_point_transform(array, self.pts_rgb)

    def get_depth(self):
        array, _ = freenect.sync_get_depth()
        array = array.astype(np.uint8)
        return four_point_transform(array, self.pts_depth)

    def ballDetect(self, curr_frame_depth, curr_frame_rgb, Ra, d):
        global centers, radii, ref_depth_for_cue

        cent_depth, rad_depth = bd.ball_det(curr_frame_depth, curr_frame_rgb, Ra, d)
        centers = cent_depth
        radii = rad_depth

        ref_depth_for_cue = self.get_depth()

        return centers, radii, ref_depth_for_cue

    def strikeDetect(self):
        global strike, ref_depth_for_cue

        if len(centers) > 0 and ref_depth_for_cue is not None and wt != 0:
            wht_center = centers[0]
            d = self.d + 20

            roi_curr = curr_frame_depth[int(wht_center[1] - d / 2):int(wht_center[1] + d / 2),
                       int(wht_center[0] - d / 2):int(wht_center[0] + d / 2)]
            roi_ref = ref_depth_for_cue[int(wht_center[1] - d / 2):int(wht_center[1] + d / 2),
                      int(wht_center[0] - d / 2):int(wht_center[0] + d / 2)]

            Sub = cv2.absdiff(roi_curr, roi_ref)
            Ts = 0.00019 * 65535
            Bin = np.zeros(Sub.shape, np.uint8)
            Bin[Sub > Ts] = 255

            Cs, _, _, _ = cv2.sumElems(Bin)

            thres_ar = 300 * np.pi * (d ** 2) / 12

            if Cs > thres_ar:
                strike = 1
            else:
                strike = 0
        strike = 0

    def cueDetect(self, curr_frame_depth, pts_depth, Qa, M, N, wht_center):
        global reflections, cue_line
        reflections, cue_line = cd.cue_det(curr_frame_depth, pts_depth, Qa, M, N, wht_center)
        return reflections, cue_line

    def run(self):
        global wt, strike, centers, reflections, cue_line, ref_depth_for_cue, curr_frame_depth
        wt, strike, centers, reflections, cue_line = 0, 0, [], [], []
        ref_depth_for_cue = None

        while 1 and isQFull:
            curr_frame_depth = self.get_depth()
            curr_frame_rgb = self.get_video()

            if wt == 0:
                reflections = []
                centers, radii, ref_depth_for_cue = self.ballDetect(curr_frame_depth, curr_frame_rgb,
                                                                    self.ref_depth_no_balls, self.d)
                strike = 0

            else:
                if strike == 0:
                    strikedet_thread = threading.Thread(target=self.strikeDetect)
                    strikedet_thread.start()
                    reflections, cue_line = self.cueDetect(curr_frame_depth, pts_depth, ref_depth_for_cue, self.M,
                                                           self.N, centers[0])

                if strike == 1:
                    centers, radii, _ = self.ballDetect(curr_frame_depth, curr_frame_rgb,
                                                        self.ref_depth_no_balls, self.d)

            print(wt, strike, centers[0], len(reflections))


if __name__ == "__main__":
    # Load Config
    with open('../sys_setup/pts_depth.pkl', 'rb') as input1:
        pts_depth = pickle.load(input1)

    with open('../sys_setup/pts_rgb.pkl', 'rb') as input2:
        pts_rgb = pickle.load(input2)

    ref_depth_no_balls = cv2.imread('../sys_setup/ref_depth_no_balls.png')
    ref_rgb_no_balls = cv2.imread('../sys_setup/ref_rgb_no_balls.png')

    motiondet_thread = MotionDetect(pts_depth)
    process_thread = Process(pts_depth, pts_rgb, ref_depth_no_balls[:, :, 0], ref_rgb_no_balls[:, :, 0])
    motiondet_thread.start()
    time.sleep(1)
    process_thread.start()
    time.sleep(1)
    app = MainApplication(pts_depth)
    app.GUI()
