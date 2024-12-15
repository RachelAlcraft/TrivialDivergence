"""
Author: Rachel ALcraft
Date 15/12/25

This class constructs a polygon object given the required number of vertices.
The shape can be even or unevenly distributed in both the angles and the spokes.

The generation of the polygon in by this method ensures a boundary can be drawn around the points
that does not intersect itself.

"""

import leuci_xyz.vectorthree as v3
import leuci_xyz.spacetransform as sp
import numpy as np
import math



class Polygon:
    def __init__(self, vertices, centre, linear, planar, point, even_spread = True):
        # if evenly spread then the points are evenly distributed around the centre
        # if unevenly spread then the points are distributed around the centre but not evenly
        self.vertices = vertices
        self.points = []
        space = sp.SpaceTransform(centre, linear, planar)
        angle_diff = 360/vertices
        angle_left = 360
        my_point = v3.VectorThree(point[0],point[1],point[2])
        for v in range(vertices):
            if even_spread:
                my_point = space.navigate(my_point,"CL",0,math.radians(angle_diff))
            else:
                new_angle = np.random.randint(1,angle_left-3*vertices+v)
                my_point = space.navigate(my_point,"CL",0,math.radians(new_angle))
                angle_left -= new_angle          
            self.points.append(my_point)            
            
            
                
        
        
        
        
        
        
        