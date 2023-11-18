'''
This test is the simplest check that every class initialises and every function runs
We are going to replicate the wikipaedia example first
https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
These are 1 dimensional tests
'''
import pandas as pd
import os
import src.ra_trivial.trivial as awa
import ReportExport as re

# Set up a report to save the results to
dir_path = os.path.dirname(os.path.realpath(__file__))
test1 = re.ReportExport('Test Set 1',dir_path + '/output/Tests01.html',cols=6)

# TESTS 1 and 2 K-L from wikipedia gives the same reults
bins = 3
test1.addLineComment(' ----- Test 01 ----- <br/>Kullback-Leibler P-Q') ####################################################################################
dataA = pd.DataFrame(data={'col1':[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2]})
dataB = pd.DataFrame(data={'col1':[0,1,2]})
rae_mark_kl_pq = awa.AlcraftWilliamsAssociation(dataA,dataB,method='k-l',bins=bins)
assocpq = rae_mark_kl_pq.addAssociation(['col1'])
statpq = round(assocpq.metric,5)
## OUTPUT TO PRINT and REPORT
# Ouput to report
test1.addPlot1d(dataA,'histogram','col1',bins=bins)
test1.addPlot1d(dataB,'histogram','col1',bins=bins)
boxA = ''
for v in assocpq.matA:
    boxA += str(round(v,5)) + '<br/>'
test1.addBoxComment(boxA)
boxD = ''
for v in assocpq.matDiff:
    boxD += str(round(v,5)) + '<br/>'
test1.addBoxComment(boxD)
boxB = ''
for v in assocpq.matB:
    boxB += str(round(v,5)) + '<br/>'
test1.addBoxComment(boxB)
# and to print
if (statpq == 0.08530):
    print('TEST 01 has passed: Kullback-Leibler Divergence PQ',statpq)
    test1.addBoxComment('TEST 01 has passed: Kullback-Leibler Divergence PQ ' + str(statpq))
else:
    print('!!! TEST 01 FAILED !!!: Kullback-Leibler Divergence PQ',statpq)
    test1.addBoxComment('!!! TEST 01 FAILED !!!: Kullback-Leibler Divergence PQ ' + str(statpq))

test1.addLineComment(' ----- Test 02 ----- <br/>Kullback-Leibler Q-P') ###################################################################################
rae_mark_kl_qp = awa.AlcraftWilliamsAssociation(dataB,dataA,method='k-l',bins=3)
assocqp = rae_mark_kl_qp.addAssociation(['col1'])
statqp = round(assocqp.metric,5)
# Ouput to report
test1.addPlot1d(dataB,'histogram','col1',bins=bins)
test1.addPlot1d(dataA,'histogram','col1',bins=bins)
boxA = ''
for v in assocqp.matA:
    boxA += str(round(v,5)) + '<br/>'
test1.addBoxComment(boxA)
boxD = ''
for v in assocqp.matDiff:
    boxD += str(round(v,5)) + '<br/>'
test1.addBoxComment(boxD)
boxB = ''
for v in assocqp.matB:
    boxB += str(round(v,5)) + '<br/>'
test1.addBoxComment(boxB)
# and to print
if (statqp == 0.09746):
    print('TEST 01 has passed: Kullback-Leibler Divergence PQ',statqp)
    test1.addBoxComment('TEST 01 has passed: Kullback-Leibler Divergence PQ ' + str(statqp))
else:
    print('!!! TEST 01 FAILED !!!: Kullback-Leibler Divergence PQ',statpq)
    test1.addBoxComment('!!! TEST 01 FAILED !!!: Kullback-Leibler Divergence PQ ' + str(statqp))

# TESTS 3 my abs diff calc
test1.addLineComment(' ----- Test 03 ----- <br/>AbsVal Diffference calculation') ###################################################################################
rae_mark = awa.AlcraftWilliamsAssociation(dataA,dataB,bins=3)
assoc = rae_mark.addAssociation(['col1'])
stat = round(assoc.metric,5)
# Ouput to report
test1.addPlot1d(dataB,'histogram','col1',bins=bins)
test1.addPlot1d(dataA,'histogram','col1',bins=bins)
boxA = ''
for v in assoc.matA:
    boxA += str(round(v,5)) + '<br/>'
test1.addBoxComment(boxA)
boxD = ''
for v in assoc.matDiff:
    boxD += str(round(v,5)) + '<br/>'
test1.addBoxComment(boxD)
boxB = ''
for v in assoc.matB:
    boxB += str(round(v,5)) + '<br/>'
test1.addBoxComment(boxB)
# and to print
if (stat == 0.17333):
    print('TEST 03 has passed: Abs Dfference',stat)
    test1.addBoxComment('TEST 03 has passed: Abs Dfference ' + str(stat))
else:
    print('!!! TEST 03 has FAILED!!!: Abs Difference',stat)
    test1.addBoxComment('!!! TEST 03 has FAILED !!!: Abs Difference ' + str(stat))

test1.addLineComment(' ----- Test 04 ----- <br/>Compare 1 histogram to convolved') ###################################################################################
rae_mark = awa.AlcraftWilliamsAssociation(dataA,bins=3)
assoc = rae_mark.addAssociation(['col1'])
stat = round(assoc.metric,5)
# Ouput to report
test1.addPlot1d(dataB,'histogram','col1',bins=bins)
test1.addPlot1d(dataA,'histogram','col1',bins=bins)
boxA = ''
for v in assoc.matA:
    boxA += str(round(v,5)) + '<br/>'
test1.addBoxComment(boxA)
boxD = ''
for v in assoc.matDiff:
    boxD += str(round(v,5)) + '<br/>'
test1.addBoxComment(boxD)
boxB = ''
for v in assoc.matB:
    boxB += str(round(v,5)) + '<br/>'
test1.addBoxComment(boxB)
# and to print
if (stat == 0):
    print('TEST 05 has passed: Kullback-Leibler Divergence Zero',stat)
    test1.addBoxComment('TEST 05 has passed: K-L zero ' + str(stat))
else:
    print('!!! TEST 05 FAILED !!!: Kullback-Leibler Divergence Zero',stat)
    test1.addBoxComment('!!! TEST 05 has FAILED !!!: K-L zero ' + str(stat))

# Finally print out the report
test1.printReport()

