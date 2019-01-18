import Tkinter

main = Tkinter.Tk()

def create_circle(x, y, r, canvasName): #center coordinates, radius
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return canvasName.create_oval(x0, y0, x1, y1)

##INPUTS
#x - x coordinate of white ball center
#y - y coordinate of white ball center
#rad - radius of white ball
#p1 - first point describing the cue, list of x and y coordinate
#q1 - first point in the wall, list of x and y coordinate
#q2
#q3
#breadth - smaller side length
# class
def visualFunc(x, y, rad, p1, q1, q2, q3, breadth):
    V1x, V1y = 10, 10
    V4x, V4y = 310, 610

    canvas_width = 1000
    canvas_height = 1000

    canvas = Tkinter.Canvas(main, width = canvas_width, height = canvas_height)
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

    #white ball circle
    white_ball_rad = rad + rad/breadth
    create_circle(x, y, rad, canvas)

    #Line 1
    canvas.create_line(p1[0], p1[-1], q1[0], q1[-1], fill="blue")

    #Line 2
    canvas.create_line(q1[0], q1[-1], q2[0], q2[-1], fill="blue")

    #Line 3
    canvas.create_line(q2[0], q2[-1], q3[0], q3[-1], fill="blue")

    canvas.pack()
    main.mainloop()


#Sample inputs
x = 35
y = 128
rad = 29
p1 = [56, 224]
q1 = [123, 232]
q2 = [565, 341]
q3 = [123, 543]
breadth = 500

#Test call actually not needed
visualFunc(x, y, rad, p1, q1, q2, q3, breadth)
