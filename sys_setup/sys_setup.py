import cv2
import freenect
import numpy as np
from tkinter import *
import pickle
from cameradata.utils.get_pts_gui import get_points, get_depth, get_video
from cameradata.utils.perspTransform import four_point_transform



class Window(Frame):

    def __init__(self, master=None):
        Frame.__init__(self, master, background="blue")
        self.master = master
        self.init_window()
        self.pts_rgb = None
        self.pts_depth = None
        self.ref_depth_no_balls = None
        self.ref_rgb_no_balls = None


    def init_window(self):
        self.master.title("Welcome Screen")

        self.pack(fill=BOTH, expand=1)

        T = Text(self, height=1, width=27)
        T.pack(fill=BOTH, expand=2)
        T.tag_configure('bold_italics', font=('Arial', 13, 'bold', 'italic'))
        T.insert(END, "WELCOME TO BILLIARDS", "bold_italics")
        T.place(x=80, y=120)

        calibrate_button = Button(self, text="Calibrate", command=self.califunc)
        calibrate_button.place(x=200, y=250)

        load_prev = Button(self, text="Load Prev Config")
        load_prev.place(x=280, y=250)


    def califunc(self):
        d = Tk()
        frame2 = Frame(d, height=100, width=300, bg="black")
        frame2.pack(fill=BOTH, expand=1)

        auto_button = Button(frame2, text="Auto", command=self.auto_calib)
        auto_button.place(x=80, y=40)
        manual_button = Button(frame2, text="Manual", command=self.man_calib)
        manual_button.place(x=190, y=40)



    def man_calib(self):
        self.pts_rgb = get_points(0)
        self.pts_depth = get_points(1)
        self.ref_depth_no_balls = four_point_transform(get_depth(), self.pts_depth)
        self.ref_rgb_no_balls = four_point_transform(get_video(), self.pts_rgb)
        with open('pts_rgb.pkl', 'wb') as output1:
            pickle.dump(self.pts_rgb, output1)

        with open('pts_depth.pkl', 'wb') as output2:
            pickle.dump(self.pts_depth, output2)

        cv2.imwrite('ref_depth_no_balls.png', self.ref_depth_no_balls)
        cv2.imwrite('ref_rgb_no_balls.png', self.ref_rgb_no_balls)

        self.quit()

    def auto_calib(self):
        pass



if __name__ == "__main__":
    root = Tk()
    root.geometry("500x300")
    app = Window(root)
    root.mainloop()