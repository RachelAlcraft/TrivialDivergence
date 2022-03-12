'''
This test is to check the p-value
2 d tests
'''
from math import cos
import numpy as np
import pandas as pd
import os
import AlcraftWilliamsAssociation as awa
import ReportExport as re

# Set up a report to save the results to
dir_path = os.path.dirname(os.path.realpath(__file__))
test3 = re.ReportExport('Test Set 3',dir_path + '/output/Tests03.html',cols=6)

bins = 20
def addTest(num,datas,cols,comment,result):
    test3.addLineComment(comment) ###################################################################################
    if len(datas) == 1:
        dataA = datas[0]
        rae_mark = awa.AlcraftWilliamsAssociation(dataA,bins=bins,piters=1000)
        dataB =rae_mark.getShuffledData(dataA ,cols)
    else:
        dataA=datas[0]
        dataB=datas[1]
        rae_mark = awa.AlcraftWilliamsAssociation(dataA,dataB,bins=bins,piters=1000)
            
    assoc = rae_mark.addAssociation(['col1','col2'])
    stat = round(assoc.metric,5)
    # Ouput to report
    A = assoc.matA
    B = assoc.matB
    D = assoc.matDiff
    hA = assoc.phistA
    hB = assoc.phistB    
    pvalue = assoc.pvalue
    test3.addPlot2d(dataA,'scatter',geo_x='col1',geo_y='col2')    
    test3.addPlot2d(dataB,'scatter',geo_x='col1',geo_y='col2')
    maxV = max(np.max(D), -1*np.min(D))
    #test3.addSurface(A, 'Original Data', cmin=0, cmax=maxV, palette='Blues', colourbar=False)    
    #test3.addSurface(B, 'Compare Data', cmin=0, cmax=maxV, palette='Reds', colourbar=False)    
    test3.addPlot1d(hA, 'histogram', title='', overlay=True, alpha=0.5,palette='steelblue')
    test3.addPlot1d(hB, 'histogram', title='', alpha=0.5,palette='Firebrick')
    test3.addSurface(D, 'Difference Data', cmin=-1 * maxV, cmax=maxV, palette='RdBu', colourbar=False)

    # print results
    passes = False
    if len(result) == 1:
        passes = stat == result
    else:
        passes = stat >= result[0] and stat <= result[1]
    if passes:
        print('TEST',num,'has passed',stat)
        test3.addBoxComment('TEST 01 has passed<br/>Metric=' + str(stat) + '<br/>P-value=' + str(round(pvalue,4)))
    else:
        print('!!! TEST ',num, 'FAILED !!!',stat)
        test3.addBoxComment('!!! TEST ' + str(num) + ' FAILED !!!<br/>Metric=' + str(stat) + '<br/>P-value=' + str(round(pvalue,4)))

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

addTest(1,[dataA],['col1','col2'],' ----- Test 01 ----- <br/>Highly associated line',[0.95])
addTest(2,[dataB],['col1','col2'],' ----- Test 02 ----- <br/>Quite random',[0.3,0.6])
addTest(3,[dataC],['col1','col2'],' ----- Test 03 ----- <br/>Sinusoidal',[0.80414])
addTest(4,[dataD],['col1','col2'],' ----- Test 04 ----- <br/>Blurred line',[0.6,0.8])
addTest(5,[dataA,dataD],['col1','col2'],' ----- Test 05 ----- <br/>Line to line',[0.3,0.6])

# Finally print out the report
test3.printReport()



