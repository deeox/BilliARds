from cameradata.ball_detect.ball_detect import *

pts_depth = get_points(1)
pts_rgb = get_points(0)
Ra, Ra_rgb = get_ball_reference(pts_depth, pts_rgb)

sleep(5)

