import freenect
import pickle
import queue
import sys
import threading
import time
import tkinter as tk
import os

import cv2
import numpy as np

import cameradata.ball_detect.ball_detect as bd
import cameradata.cue_detect.cue_detect as cd
from cameradata.utils.perspTransform import four_point_transform

global wt, strike, centers, reflections, cue_line

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        global wt, strike, centers, reflections, cue_line
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent
        self.canvas = tk.Canvas(self.parent, background="black")

        self.root_2_times_2 = 2.828
        self.scale = 2
        self.R = 6
        self.M = 387 * self.scale
        self.N = self.M / 2
        self.thickness = 5
        self.circle_countour_scale = 1.5
        self.offset = 10


    def GUI(self):

        M = self.M
        N = self.N
        offset = self.offset

        self.canvas.pack(fill=tk.BOTH, expand=True)
        self.canvas.create_rectangle(0 + offset, 0 + offset, M + offset, N + offset, width=1, outline="white")

        self.draw_balls()
        self.draw_traj()


        self.parent.attributes("-fullscreen", True)
        self.parent.bind('c', self.quitApp)
        self.parent.bind('<Escape>', self.quitApp)

    def draw_balls(self):
        for i in range(len(centers)):
            root_2_times_2 = self.root_2_times_2
            R = self.R
            offset = self.offset
            circle_countour_scale = self.circle_countour_scale
            self.canvas.create_oval(centers[i][0] - root_2_times_2 * circle_countour_scale * R + offset,
                               centers[i][1] - root_2_times_2 * circle_countour_scale * R + offset,
                               centers[i][0] + root_2_times_2 * circle_countour_scale * R + offset,
                               centers[i][1] + root_2_times_2 * circle_countour_scale * R + offset, fill="white", )
        self.update()
        self.canvas.pack()
        self.after(50, self.draw_balls)

    def draw_traj(self):
        for i in range(len(reflections) - 1):
            root_2_times_2 = self.root_2_times_2
            offset = self.offset
            R = self.R
            thickness = self.thickness
            self.canvas.create_line(reflections[i][0] + offset, reflections[i][1] + offset, reflections[i + 1][0] + offset,
                               reflections[i + 1][1] + offset, fill="blue", width=thickness)
            self.canvas.create_oval(reflections[i + 1][0] - root_2_times_2 * R + offset,
                               reflections[i + 1][1] - root_2_times_2 * R + offset,
                               reflections[i + 1][0] + root_2_times_2 * R + offset,
                               reflections[i + 1][1] + root_2_times_2 * R + offset, fill="blue", )

        #self.canvas.create_oval(reflections[-1][0] - root_2_times_2 * R + offset,
        #                   reflections[-1][1] - root_2_times_2 * R + offset,
        #                   reflections[-1][0] + root_2_times_2 * R + offset,
        #                   reflections[-1][1] + root_2_times_2 * R + offset, fill="blue", )
        self.update()
        self.canvas.pack()
        self.after(50, self.draw_traj)


    @staticmethod
    def quitApp(event=None):
        os._exit(0)


class MotionDetect(threading.Thread):
    def __init__(self, pts_depth):
        threading.Thread.__init__(self)

        self.pts_depth = pts_depth
        self.Np = 5
        self.imageQ = queue.Queue(maxsize=self.Np)
        self.wt = 0  # 0 is no motion, 1 is motion at t
        self.wt_1 = 0  # 0 is no motion, 1 is motion at t_1
        self.Kt = 0  # Counter
        self.K1 = 250  # 0.7 * M
        self.K2 = 2

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
        K2 = self.K2
        Np = self.Np

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

            Tm = 0.0014 * 65535
            motionBin = np.zeros(motionMat.shape)
            motionBin[motionMat >= Tm] = 1

            Cmt, _, _, _ = cv2.sumElems(motionBin)
            # print(Cmt)

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

            self.imageQ.get()
            self.imageQ.put(self.get_depth())

            wt_1 = wt


class Process(threading.Thread):
    def __init__(self, pts_depth, pts_rgb, ref_depth_no_balls, ref_rgb_no_balls):
        threading.Thread.__init__(self)

        self.pts_depth = pts_depth
        self.pts_rgb = pts_rgb
        self.ref_depth_no_balls = ref_depth_no_balls
        self.ref_rgb_no_balls = ref_rgb_no_balls
        self.curr_frame_depth = self.get_depth()
        self.curr_frame_rgb = self.get_video()
        self.M = int(self.pts_depth[1][0] - self.pts_depth[0][0])
        self.N = int(self.pts_depth[2][0] - self.pts_depth[0][0])


    def get_video(self):
        array, _ = freenect.sync_get_video()
        array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)
        return four_point_transform(array, self.pts_rgb)

    def get_depth(self):
        array, _ = freenect.sync_get_depth()
        array = array.astype(np.uint8)
        return four_point_transform(array, self.pts_depth)

    def ballDetect(self, curr_frame_depth, curr_frame_rgb, Ra):
        global centers, radii, ref_depth_for_cue

        # curr_frame_depth, curr_frame_rgb, Ra = self.curr_frame_depth, self.curr_frame_rgb, self.curr_frame_depth[:,:,0]
        cent_depth, rad_depth = bd.ball_det(curr_frame_depth, curr_frame_rgb, Ra)
        centers = cent_depth
        radii = rad_depth

        ref_depth_for_cue = self.get_depth()
        # print(len(cent_depth))
        # print(cent_depth[0])
        cv2.destroyAllWindows()
        return centers, radii, ref_depth_for_cue

    def strikeDetect(self):
        global strike

        strike = 0

        #print(strike)

    def cueDetect(self, curr_frame_depth, pts_depth, Qa, M, N, wht_center):
        global reflections, cue_line
        #reflections, cue_line = [], []

        reflections, cue_line = cd.cue_det(curr_frame_depth, pts_depth, Qa, M, N, wht_center)

        return reflections, cue_line

    def run(self):
        global wt, strike, centers, reflections, cue_line
        wt, strike, centers, reflections, cue_line = 0, 0, [], [], []
        ref_depth_for_cue = None

        while 1 and isQFull:
            curr_frame_depth, curr_frame_rgb = self.get_depth(), self.get_video()

            if wt == 0:
                reflections = []
                centers, radii, ref_depth_for_cue = self.ballDetect(curr_frame_depth, curr_frame_rgb,
                                                                    self.ref_depth_no_balls)
                strike = 0

            else:
                if strike == 0:
                    strikedet_thread = threading.Thread(target=self.strikeDetect)
                    strikedet_thread.start()
                    reflections, cue_line = self.cueDetect(curr_frame_depth, pts_depth, ref_depth_for_cue, self.M, self.N, centers[0])

                if strike == 1:
                    centers, radii, _ = self.ballDetect(curr_frame_depth, curr_frame_rgb,
                                                        self.ref_depth_no_balls)

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

    root = tk.Tk()
    app = MainApplication(root)
    app.GUI()
    root.mainloop()
