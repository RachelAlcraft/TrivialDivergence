'''
This test is to check if the seach for multi dim association to a col works
'''
from math import cos
import numpy as np
import pandas as pd
import os
import src.ra_trivial.trivial as awa
import ReportExport as re

# Set up a report to save the results to
dir_path = os.path.dirname(os.path.realpath(__file__))
test7 = re.ReportExport('Test Set 7',dir_path + '/output/Tests07.html',cols=4)


# now generate 4 sets of more extremme data
x = []
lineA = []
lineB = []
lineC = []
lineD = []
lineE = []

bins = 5
method = 'diff'
piters = 5

for i in range(1000):
    x.append(i)
    lineA.append(i)    
    lineB.append(np.random.normal(0,1000))
    lineC.append(cos(i/100))
    lineD.append(i + np.random.normal(0,50))
    lineE.append(i+ np.random.normal(0,50))

dataA = pd.DataFrame(data={'col1':x,'col2':lineA,'col3':lineB,'col4':lineC,'col5':lineD,'col6':lineE})
rae_mark = awa.AlcraftWilliamsAssociation(dataA,bins=bins,piters=piters,method=method,loglevel=2)


df1 = rae_mark.getStrongestAssociations([],dataA.columns,2,0.5)
df2 = rae_mark.getStrongestAssociations(['col1'],dataA.columns,1,0.5)
df3 = rae_mark.getStrongestAssociations(['col1'],dataA.columns,2,1)
df4 = rae_mark.getStrongestAssociations(['col1','col2'],dataA.columns,1,1)

print('2d all search')
print(df1)
print('2d with col1')
print(df2)
print('3d with col1')
print(df3)
print('3d with col1,col2')
print(df4)

# Finally print out the report
test7.printAssociations(rae_mark,df3,'col1','col2','col3')



