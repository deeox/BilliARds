from tkinter import *

class Window(Frame):
    
    def __init__(self, master=None):
        Frame.__init__(self,master, background= "blue")
        self.master = master
        self.init_window()
        
    def init_window(self):
        
        self.master.title("Welcome Screen")
        
        self.pack(fill=BOTH, expand=1)
        
        T = Text(self, height= 1, width=27)
        T.pack(fill=BOTH, expand=2)
        T.tag_configure('bold_italics', font=('Arial', 13, 'bold', 'italic'))
        T.insert(END, "WELCOME TO BILLIARDS","bold_italics")
        T.place(x=80,y=120)
        
        calibrate_button =  Button(self, text = "Calibrate", command = self.califunc)
        calibrate_button.place(x=200,y=250)
        
        load_prev =  Button(self, text = "Load Prev Config")
        load_prev.place(x=280, y=250)
        
        
    def califunc(self):
        
        d = Tk()
        frame2 = Frame(d, height=100, width= 300, bg="black")
        frame2.pack(fill=BOTH, expand=1)
       
        auto_button = Button(frame2, text="Auto")
        auto_button.place(x=80,y=40)
        manual_button = Button(frame2, text= "Manual")
        manual_button.place(x=190,y=40)


root = Tk()
root.geometry("400x300")
app = Window(root)
root.mainloop()
