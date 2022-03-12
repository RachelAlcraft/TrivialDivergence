'''
This test is the simplest check that every class initialises and every function runs
We are going to replicate the wikipaedia example first
https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence
'''
import pandas as pd

import AlcraftWilliamsAssociation as awa

dataA = pd.DataFrame(data={'col1':[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,2,2,2,2]})
dataB = pd.DataFrame(data={'col1':[0,1,2]})

# TESTS 1 and 2 K-L from wikipedia gives the same reults
rae_mark_kl_pq = awa.AlcraftWilliamsAssociation(dataA,dataB,method='k-l',bins=3)
assocpq = rae_mark_kl_pq.addAssociation(['col1'])
statpq = round(assocpq.metric,5)
rae_mark_kl_qp = awa.AlcraftWilliamsAssociation(dataB,dataA,method='k-l',bins=3)
assocqp = rae_mark_kl_qp.addAssociation(['col1'])
statqp = round(assocqp.metric,5)
if (statpq == 0.08530):
    print('TEST 01 has passed: Kullback-Leibler Divergence PQ',statpq)
else:
    print('!!! TEST 01 FAILED !!!: Kullback-Leibler Divergence PQ',statpq)
if (statqp == 0.09746):
    print('TEST 02 has passed: Kullback-Leibler Divergence QP',statqp)
else:
    print('!!! TEST 02 FAILED !!!: Kullback-Leibler Divergence QP',statqp)


# TESTS 3 and 4 my abs diff calc
rae_mark_kl_pq = awa.AlcraftWilliamsAssociation(dataA,dataB,bins=3)
assocpq = rae_mark_kl_pq.addAssociation(['col1'])
statpq = round(assocpq.metric,5)
rae_mark_kl_qp = awa.AlcraftWilliamsAssociation(dataB,dataA,bins=3)
assocqp = rae_mark_kl_qp.addAssociation(['col1'])
statqp = round(assocqp.metric,5)
if (statpq == 0.26):
    print('TEST 03 has passed: Abs Difference Divergence PQ',statpq)
else:
    print('!!! TEST 03 FAILED !!!: Abs Difference Divergence PQ',statpq)
if (statqp == 0.26):
    print('TEST 04 has passed: Abs Difference Divergence QP',statqp)
else:
    print('!!! TEST 04 FAILED !!!: Abs Difference Divergence QP',statqp)

# TEST 5 K-L from wikipedia convolved is 0
rae_mark_kl_pq = awa.AlcraftWilliamsAssociation(dataA,method='k-l',bins=3)
assocpq = rae_mark_kl_pq.addAssociation(['col1'])
statpq = round(assocpq.metric,5)
if (statpq == 0):
    print('TEST 05 has passed: Kullback-Leibler Divergence Zero',statpq)
else:
    print('!!! TEST 05 FAILED !!!: Kullback-Leibler Divergence Zero',statpq)

# TEST 6 Make a simple 2d case
dataA = pd.DataFrame(data={'col1':[0,1,2],'col2':[0,1,2]})
rae_mark_kl_pq = awa.AlcraftWilliamsAssociation(dataA,bins=3)
assocpq = rae_mark_kl_pq.addAssociation(['col1','col2'])
statpq = round(assocpq.metric,5)
if (statpq == 1):
    print('TEST 06 has passed: Kullback-Leibler Divergence Straight Line',statpq)
else:
    print('!!! TEST 06 FAILED !!!: Kullback-Leibler Divergence Straight Line',statpq)

# TEST 7 Make a simple 2d case
dataA = pd.DataFrame(data={'col1':[0,1,2,3,4,5],'col2':[0,1,2,3,4,5]})
rae_mark_kl_pq = awa.AlcraftWilliamsAssociation(dataA,bins=6)
assocpq = rae_mark_kl_pq.addAssociation(['col1','col2'])
statpq = round(assocpq.metric,5)
if (statpq == 1):
    print('TEST 07 has passed: Kullback-Leibler Divergence Straight Line',statpq)
else:
    print('!!! TEST 07 FAILED !!!: Kullback-Leibler Divergence Straight Line',statpq)

# TEST 8 Make a simple 2d case
dataA = pd.DataFrame(data={'col1':[0,1,2,3,3,3],'col2':[3,3,5,5,4,4]})
rae_mark_kl_pq = awa.AlcraftWilliamsAssociation(dataA,bins=3)
assocpq = rae_mark_kl_pq.addAssociation(['col1','col2'])
statpq = round(assocpq.metric,5)
if (statpq == 0.75):
    print('TEST 08 has passed: Kullback-Leibler Divergence some order',statpq)
else:
    print('!!! TEST 08 FAILED !!!: Kullback-Leibler Divergence some order',statpq)

# TEST 9 Make a simple 2d case
dataA = pd.DataFrame(data={'col1':[0,1,2,3,4,5],'col2':[0,1,2,3,4,5]})
dataB = pd.DataFrame(data={'col1':[0,1,2,3,3,3],'col2':[3,3,5,5,4,4]})
rae_mark_kl_pq = awa.AlcraftWilliamsAssociation(dataA,dataB,bins=3)
assocpq = rae_mark_kl_pq.addAssociation(['col1','col2'])
statpq = round(assocpq.metric,5)
if (statpq == 0.75):
    print('TEST 09 has passed: Kullback-Leibler Divergence some order',statpq)
else:
    print('!!! TEST 09 FAILED !!!: Kullback-Leibler Divergence some order',statpq)

#rae_mark_comp.addAssociation(['col1','col1'])
#rae_mark_comp.addAssociation(['col1','col1','col1'])

#rae_mark_conv = awa.AlcraftWilliamsAssociation(dataA)
#rae_mark_conv.addAssociation(['col1'])
#rae_mark_conv.addAssociation(['col1','col1'])
#rae_mark_conv.addAssociation(['col1','col1','col1'])

