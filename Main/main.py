from threading import Thread
import tkinter as tk
import sys, pickle
import freenect
import cv2
import numpy as np
import imutils
from operator import itemgetter


class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.parent = parent

        V1x, V1y = 10, 10
        V4x, V4y = 610, 310

        canvas = tk.Canvas(self.parent, background="black")
        canvas.pack(fill=tk.BOTH, expand=True)

        canvas.create_rectangle(V1x, V1y, V4x, V4y, width=3, outline="white")

        # right-top-hole
        canvas.create_arc(585, -9, 635, 32, width=3, start=180, outline="white")

        # right-bottom-hole
        canvas.create_arc(585, 285, 635, 335, width=3, start=90, outline="white")

        # top-left-hole
        canvas.create_arc(-10, 30, 30, -10, width=3, start=270, outline="white")
        # mid-top-hole
        canvas.create_arc(290, -10, 330, 30, width=3, start=180, extent=180, outline="white")
        # bottom-left-hole
        canvas.create_arc(-9, 290, 32, 330, width=3, start=0, outline="white")

        # mid-bottom-hole
        canvas.create_arc(290, 290, 324, 330, width=3, start=0, extent=180, outline="white")

        canvas.pack()
        self.parent.attributes("-fullscreen", True)
        self.parent.bind('c', sys.exit)
        self.parent.bind('<Escape>', sys.exit)


class MotionDetect(Thread):
    def run(self):





if __name__ == "__main__":

    #Load Config
    with open('../sys_setup/pts_depth.pkl', 'rb') as input1:
        pts_depth = pickle.load(input1)

    with open('../sys_setup/pts_rgb.pkl', 'rb') as input2:
        pts_rgb = pickle.load(input2)

    ref_depth_no_balls = cv2.imread('../sys_setup/ref_depth_no_balls.png')
    ref_rgb_no_balls = cv2.imread('../sys_setup/ref_rgb_no_balls.png')




    while 1:
        root = tk.Tk()
        app = MainApplication(root)
        root.mainloop()





