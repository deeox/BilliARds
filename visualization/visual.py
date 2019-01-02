import Tkinter

V1x, V1y = 10, 10
V4x, V4y = 310, 610

canvas_width = 1000
canvas_height = 1000

top = Tkinter.Tk()
canvas = Tkinter.Canvas(top, width = canvas_width, height = canvas_height)
main_table = canvas.create_rectangle(V1x, V1y, V4x, V4y, width=3)

#bottom-left-hole
canvas.create_arc(-10, 590, 30, 630, width=3)
#bottom-right-hole
canvas.create_arc(330, 590, 290, 630,width=3, start=90)
#top-left-hole
canvas.create_arc(-10, 30, 30, -10, width=3, start=270)
#top-right-hole
canvas.create_arc(290, -10, 330, 30, width=3, start=180)
#mid-left-hole
canvas.create_arc(-2, 305, 25, 340, width=3, start=270, extent=180)

#mid-right-hole
canvas.create_arc(297, 305, 324, 340, width=3, start = 90, extent=180)

#canvas.create_oval(310, 590, 310, 590, width = 10, fill = 'red')

canvas.pack()
top.title("BilliARds")
top.mainloop()
