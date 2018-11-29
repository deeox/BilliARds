import numpy as np

class Ball:
	
	elasticity = 0.95
	
	def __init__(self, x_pos, y_pos, radius, colour, moving = False):
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
		self.diameter = 2*radius
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
			return self.x_pos,self.x_pos
				
	def wall_collision(self, table_length, table_breadth):
		"""
		Returns the position of center of the ball when it's about to collide with the wall which will be used to plot projected path
		To get multiple collisions, just update x_vec and y_vec with (x_vec, -1*y_vec) or (-1*x_vec,y_vec) based on which wall it collides with
		"""
		
		R = self.radius
		#For 90degree shots
		if( self.x_vec == 0 and self.y_vec>0 ):
			return self.x_pos,table_breadth-R
		
		#For 270degree shots
		elif( self.x_vec == 0 and self.y_vec<0 ):
			return self.x_pos,R
		
		#For 0degree shots
		elif( self.x_vec > 0 and self.y_vec == 0 ):
			return table_length-R,self.y_pos
		
		#For 180degree shots
		elif( self.x_vec < 0 and self.y_vec == 0 ):
			return R,self.y_pos
		#Other angles
		else:
			slope = (self.y_vec)/(self.x_vec)
			# y- y_ball = slop* (x - x_ball)
			#Equations where x,y correspond to a particular wall boundary where collision takes place based on https://bit.ly/2Ab1NdR 
			#Ball moves towards top right
			if( self.x_vec>=0 and self.y_vec>=0 ):
				#Check if it collides with the top or right side
				
				#Collides with right side at co-ordinates (table_length-R,y)
				y = self.y_pos + (slope)*(table_length-R - self.x_pos)
				#Collides with top side at co-ordinates (x,table_breadth-R)
				x = self.x_pos + (1/slope)*(table_breadth - R - self.y_pos)
				
				if ( y<= table_breadth-R):
					return table_length-R,y
				
				elif ( x<= table_length-R):
					return x,table_breadth-R

			elif( self.x_vec>=0 and self.y_vec<=0 ):				
				#Check if it collides with the bottom or right side
				
				#Collides with right side at co-ordinates (table_length-R,y)
				y = self.y_pos + (slope)*(table_length-R - self.x_pos)
				#Collides with bottom side at co-ordinates (x,R)
				x = self.x_pos + (1/slope)*(R - self.y_pos)
				
				if ( y<= table_breadth):
					return table_length-R,y
				
				elif ( x>= R):
					return x,R				

			elif( self.x_vec<=0 and self.y_vec<=0 ):
				
				#Check if it collides with the bottom or left side
				
				#Collides with left side at co-ordinates (R,y)
				y = self.y_pos + (slope)*(R - self.x_pos)
				#Collides with bottom side at co-ordinates (x,R)
				x = self.x_pos + (1/slope)*(R - self.y_pos)
				
				if ( y>= R):
					return R,y
				
				elif ( x>= R):
					return x,R				

			else:
				#Check if it collides with the top or left side
				
				#Collides with left side at co-ordinates (R,y)
				y = self.y_pos + (slope)*(R - self.x_pos)
				#Collides with top side at co-ordinates (x,table_breadth - R)
				x = self.x_pos + (1/slope)*(table_breadth - R - self.y_pos)
				
				if ( y>= R):
					return R,y
				
				elif (x<= table_length-R ):
					return x,table_breadth - R

	def ball_collision(self, second_ball):
		#TODO
		pass
		
	