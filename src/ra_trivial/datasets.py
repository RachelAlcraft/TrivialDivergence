import math
import random

import pandas as pd
import numpy as np

#############################################################################################################
### DATA CLASS ###
#############################################################################################################
class DataSetShapes:
    def __init__(self, shapes, samples = 100, even_spread = True, noise = 0):
        self.shapes = shapes
        self.samples = samples
        self.even_spread = even_spread
        self.noise = noise

    ##### Public class interface ###################################################################################    
    def getAnonDataFrame(self,cols):
        pass

    def getIdentifiableDataFrame(self,cols):
        pass
    
    ##### Private class interface ###################################################################################    