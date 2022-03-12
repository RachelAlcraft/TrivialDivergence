'''
This test is a 2d check of the associatoin, simple data reproduceable by hand
'''
from math import cos
import numpy as np
import pandas as pd
import os
import AlcraftWilliamsAssociation as awa
import ReportExport as re

# Set up a report to save the results to
dir_path = os.path.dirname(os.path.realpath(__file__))
test2 = re.ReportExport('Test Set 2',dir_path + '/output/Tests02.html',cols=6)

bins = 20
def addTest(num,datas,cols,comment,result):
    test2.addLineComment(comment) ###################################################################################
    if len(datas) == 1:
        dataA = datas[0]
        rae_mark = awa.AlcraftWilliamsAssociation(dataA,bins=bins)
        dataB =rae_mark.getShuffledData(dataA ,cols)
    else:
        dataA=datas[0]
        dataB=datas[1]
        rae_mark = awa.AlcraftWilliamsAssociation(dataA,dataB,bins=bins)
            
    assoc = rae_mark.addAssociation(['col1','col2'])
    stat = round(assoc.metric,5)
    # Ouput to report
    A = assoc.matA
    B = assoc.matB
    D = assoc.matDiff
    test2.addPlot2d(dataA,'scatter',geo_x='col1',geo_y='col2')    
    test2.addPlot2d(dataB,'scatter',geo_x='col1',geo_y='col2')
    maxV = max(np.max(D), -1*np.min(D))
    test2.addSurface(A, 'Original Data', palette='Blues', colourbar=False)
    test2.addSurface(D, 'Difference Data', cmin=-1 * maxV, cmax=maxV, palette='RdBu', colourbar=False)
    test2.addSurface(B, 'Compare Data', palette='Reds', colourbar=False)
    # print results
    passes = False
    if len(result) == 1:
        passes = stat == result
    else:
        passes = stat >= result[0] and stat <= result[1]
    if passes:
        print('TEST',num,'has passed',stat)
        test2.addBoxComment('TEST 01 has passed<br/>' + str(stat))
    else:
        print('!!! TEST ',num, 'FAILED !!!',stat)
        test2.addBoxComment('!!! TEST ' + str(num) + ' FAILED !!!<br/>' + str(stat))

# TEST 1 Make a simple 2d case of a line which is highly correlated
dataA = pd.DataFrame(data={'col1':[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4],'col2':[0,0,0,1,1,1,2,2,2,3,3,3,4,4,4]})
comment = ' ----- Test 01 ----- <br/>Highly associated line'
addTest(1,[dataA],['col1','col2'],comment,[0.8])

# now generate 4 sets of more extremme data
x = []
lineA = []
lineB = []
lineC = []
lineD = []

for i in range(1000):
    x.append(i)
    lineA.append(i)    
    lineB.append(np.random.normal(0,1000))
    lineC.append(cos(i/100))
    lineD.append(i + np.random.normal(0,50))

dataA = pd.DataFrame(data={'col1':x,'col2':lineA})
dataB = pd.DataFrame(data={'col1':x,'col2':lineB})
dataC = pd.DataFrame(data={'col1':x,'col2':lineC})
dataD = pd.DataFrame(data={'col1':x,'col2':lineD})

addTest(2,[dataA],['col1','col2'],' ----- Test 02 ----- <br/>Highly associated line',[0.95])
addTest(3,[dataB],['col1','col2'],' ----- Test 03 ----- <br/>Quite random',[0.3,0.6])
addTest(4,[dataC],['col1','col2'],' ----- Test 04 ----- <br/>Sinusoidal',[0.80414])
addTest(5,[dataD],['col1','col2'],' ----- Test 05 ----- <br/>Blurred line',[0.6,0.8])
addTest(6,[dataA,dataD],['col1','col2'],' ----- Test 06 ----- <br/>Line to line',[0.3,0.6])

# Finally print out the report
test2.printReport()

