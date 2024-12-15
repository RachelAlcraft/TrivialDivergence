import math
import random

import pandas as pd
import numpy as np
import leuci_xyz.vectorthree as v3
import polygon as pg

#############################################################################################################
### DATA CLASS ###
#############################################################################################################
class DataSetShapes:
    def __init__(self, vertices, dims = 2, samples = 100, grid_size = 100, even_spread = True, noise = 0):
        self.vertices = vertices
        self.dims = dims
        self.samples = samples
        self.grid_size = grid_size
        self.even_spread = even_spread
        self.noise = noise
        self.generateData()

    ##### Public class interface ###################################################################################    
    def getAnonDataFrame(self):
        return self.angles_frame[['rowid','angle']]

    def getIdentifiableDataFrame(self):
        return self.angles_frame
    
    ##### Private class interface ###################################################################################
    def generateData(self):
        # First generate the coordinates of the shapes
        shapes_frame = pd.DataFrame(columns=['shape','shapeid','rowid','x','y','z'])
        angles_frame = pd.DataFrame(columns=['shape','shapeid','rowid','angle'])
        row_id = 0        
        shape_id = 0
        angle_id = 0
        for vertex in self.vertices:
            print(f"~~~~~ Generating shapes for vertices={vertex} ~~~~~")
            for i in range(self.samples):                
                points = []
                xs = np.random.randint(1,self.grid_size)
                ys = np.random.randint(1,self.grid_size)
                zs = np.random.randint(1,self.grid_size)
                if self.dims == 2:            
                    zs = np.zeros(1)
                point = (xs,ys,zs)
                centre = v3.VectorThree(0,0,0)
                linear = v3.VectorThree(0,1,0)
                planar = v3.VectorThree(1,1,0)
                poly = pg.Polygon(vertex,centre,linear,planar,point,even_spread=False)
                points = poly.points
                                                
                # Add to the data frame
                for p in range(len(points)):
                    point = points[p]
                    shapes_frame.loc[row_id] = [vertex,shape_id,row_id,point.A,point.B,point.C]
                    row_id += 1
                total_angle = 0
                for p in range(len(points)):
                    if p == len(points)-2:
                        point1 = points[p]
                        point2 = points[p+1]
                        point3 = points[0]                                       
                    elif p == len(points)-1:
                        point1 = points[p]
                        point2 = points[0]
                        point3 = points[1]                                            
                    else:
                        point1 = points[p]
                        point2 = points[p+1]
                        point3 = points[p+2]                                                                               
                    vec1 = v3.VectorThree(point1.A,point1.B,point1.C)
                    vec2 = v3.VectorThree(point2.A,point2.B,point2.C)
                    vec3 = v3.VectorThree(point3.A,point3.B,point3.C)
                    AB = v3.v3_subtract(vec2,vec1)
                    BC = v3.v3_subtract(vec2,vec3)                    
                    angle = math.degrees(AB.get_angle(BC))                    
                    angles_frame.loc[angle_id] = [vertex,shape_id,angle_id,angle]
                    angle_id += 1
                    total_angle += angle                    
                shape_id += 1
                #print(f"Generated shape {shape_id} with {vertex} vertices and total angle of {total_angle}")
                if str(total_angle) == "nan":
                    for point in points:
                        print(f"Point {point.A},{point.B},{point.C}")
                    
        
        self.shapes_frame = shapes_frame                                                                                                        
        self.angles_frame = angles_frame
        #print(f"Generated {len(self.shapes_frame)} shapes and {len(self.angles_frame)} angles")
        
        
        