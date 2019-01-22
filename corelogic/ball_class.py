import numpy as np


class Ball:
    elasticity = 0.95

    def __init__(self, x_pos, y_pos, radius, colour, moving=False):
        """
        Initialized by the Calibrate function
        x_pos, y_pos store the co-ordinates of center of the ball
        radius is constant for all balls
        colour is a string like "Red" or "Blue"
        moving is a bool which check whether the ball has stopped moving
        x_vec,y_vec store the direction of movement of the ball
        """
        self.x_pos = x_pos
        self.y_pos = y_pos
        self.radius = radius
        self.diameter = 2 * radius
        self.colour = colour
        self.moving = moving
        self.x_vec = 0
        self.y_vec = 0

    def update_position(self, new_x, new_y):
        """
        Resets the vector
        """
        self.x_pos = new_x
        self.y_pos = new_y
        self.x_vec = 0
        self.y_vec = 0
        moving = False

    def update_cue_ball_vector(self, colliding_x_vec, colliding_y_vec):
        """
        Called every time cue position changes with respect to the white ball
        New x_vec and y_vec give the projected path before first wall collision
        Returns first point from where the projected path should be displayed
        """
        if self.colour == "White":
            self.x_vec = colliding_x_vec
            self.y_vec = colliding_y_vec
            return self.x_pos, self.x_pos

    def wall_collision(self, table_length, table_breadth, slope):
        """
        Returns the position of center of the ball when it's about to collide with the wall which will be used to plot projected path
        To get multiple collisions, just update x_vec and y_vec with (x_vec, -1*y_vec) or (-1*x_vec,y_vec) based on which wall it collides with
        Last value flag determines whether it collides with x or y axis
        """
        R = self.radius
        # flag 3 is an exit condition in case cue is not in frame Redundant check
        if (self.x_vec == 0 and self.y_vec == 0):
            return self.x_pos, self.y_pos, 3
        # For 90degree shots
        if (self.x_vec == 0 and self.y_vec >= 0):
            return self.x_pos, table_breadth - R, 0
        # For 270degree shots
        elif (self.x_vec == 0 and self.y_vec <= 0):
            return self.x_pos, R, 0
        # For 0degree shots
        elif (self.x_vec >= 0 and self.y_vec == 0):
            return table_length - R, self.y_pos, 1
        # For 180degree shots
        elif (self.x_vec <= 0 and self.y_vec == 0):
            return R, self.y_pos, 1
        # Other angles
        else:
            # y- y_ball = slop* (x - x_ball)
            # Equations where x,y correspond to a particular wall boundary where collision takes place based on https://bit.ly/2Ab1NdR
            # Ball moves towards top right
            if (self.x_vec >= 0 and self.y_vec >= 0):
                # Check if it collides with the top or right side
                # Collides with right side at co-ordinates (table_length-R,y)
                y = self.y_pos + (slope) * (table_length - R - self.x_pos)
                # Collides with bottom side at co-ordinates (x,table_breadth-R)
                x = self.x_pos + (1 / slope) * (table_breadth - R - self.y_pos)
                if (y <= table_breadth - R and y >= R):
                    return table_length - R, y, 1
                elif (x <= table_length - R and x >= R):
                    return x, table_breadth - R, 0
            elif (self.x_vec >= 0 and self.y_vec <= 0):
                # Check if it collides with the top or right side
                # Collides with right side at co-ordinates (table_length-R,y)
                y = self.y_pos + (slope) * (table_length - R - self.x_pos)
                # Collides with top side at co-ordinates (x,R)
                x = self.x_pos + (1 / slope) * (R - self.y_pos)
                if (y <= table_breadth - R and y >= R):
                    return table_length - R, y, 1
                elif (x <= table_length - R and x >= R):
                    return x, R, 0
            elif (self.x_vec <= 0 and self.y_vec <= 0):
                # Check if it collides with the bottom or left side
                # Collides with left side at co-ordinates (R,y)
                y = self.y_pos + (slope) * (R - self.x_pos)
                # Collides with bottom side at co-ordinates (x,R)
                x = self.x_pos + (1 / slope) * (R - self.y_pos)
                if (y <= table_breadth - R and y >= R):
                    return R, y, 1
                elif (x <= table_length - R and x >= R):
                    return x, R, 0
            else:
                # Check if it collides with the top or left side
                # Collides with left side at co-ordinates (R,y)
                y = self.y_pos + (slope) * (R - self.x_pos)
                # Collides with top side at co-ordinates (x,table_breadth - R)
                x = self.x_pos + (1 / slope) * (table_breadth - R - self.y_pos)
                if (y <= table_breadth - R and y >= R):
                    return R, y, 1
                elif (x <= table_length - R and x >= R):
                    return x, table_breadth - R, 0
        return R, y, 1

    def ball_collision(self, second_ball):
        # TODO
        pass


def get_wall_collisions(number_of_collisions, radius, table_length, table_breadth, cue_ball_x, cue_ball_y, cue_h_prob,
                        cue_h_norm):
    collision_cord = []
    tuning_center_ball = 0.5 * radius
    if cue_h_norm and cue_h_prob is not None:
        cue_stick_x1 = cue_h_prob[0][0]
        cue_stick_y1 = cue_h_prob[0][1]
        cue_stick_x2 = cue_h_prob[1][0]
        cue_stick_y2 = cue_h_prob[1][1]
        slope = (cue_h_norm[1][1] - cue_h_norm[0][1]) / (cue_h_norm[1][0] - cue_h_norm[0][0])
        dist1 = (cue_stick_x1 - cue_ball_x) ** 2 + (cue_stick_y1 - cue_ball_y) ** 2
        dist2 = (cue_stick_x2 - cue_ball_x) ** 2 + (cue_stick_y2 - cue_ball_y) ** 2
        ball_cue_dist = ((slope * cue_ball_x - cue_ball_y) - (slope * cue_h_norm[0][0] - cue_h_norm[0][1])) / (
                    1 + (slope ** 2))
        ball_cue_dist = abs(ball_cue_dist)
        if ball_cue_dist >= tuning_center_ball:
            return []
        if (dist1 > dist2):
            pass
        else:
            temp = cue_stick_x1
            cue_stick_x1 = cue_stick_x2
            cue_stick_x2 = temp
            temp = cue_stick_y1
            cue_stick_y1 = cue_stick_y2
            cue_stick_y2 = temp
        cue_stick_xvec = cue_stick_x2 - cue_stick_x1
        cue_stick_yvec = cue_stick_y2 - cue_stick_y1
        cue_ball = Ball(cue_ball_x, cue_ball_y, radius, "White")
        cue_ball.update_cue_ball_vector(cue_stick_x2 - cue_stick_x1, cue_stick_y2 - cue_stick_y1)
        for i in range(0, number_of_collisions, 1):
            # flag is used to check if the reflection occurs along xaxis or yaxis, xaxis is 0 while yaxis is 1
            center_x, center_y, flag = cue_ball.wall_collision(table_length, table_breadth, slope)
            collision_cord.append((int(center_x), int(center_y)))
            cue_ball.update_position(center_x, center_y)
            if flag == 0:
                cue_stick_yvec = -1 * cue_stick_yvec
            else:
                cue_stick_xvec = -1 * cue_stick_xvec
            slope = -1 * slope
            cue_ball.update_cue_ball_vector(cue_stick_xvec, cue_stick_yvec)
        return collision_cord
    return []
