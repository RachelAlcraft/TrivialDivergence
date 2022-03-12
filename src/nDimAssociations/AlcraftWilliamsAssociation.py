import math
import random

import pandas as pd
import numpy as np

#############################################################################################################
### DATA CLASS ###
#############################################################################################################
class Association:
    def __init__(self,cols,histA,histDiff,histB,stat):
        self.cols = cols        
        self.matA = histA
        self.matDiff = histDiff
        self.matB = histB        
        self.metric = stat
        self.phistA = None
        self.phistB = None
        self.pvalue = -1


###################################################################################################################
### ASSOCIATION CLASS ###
###################################################################################################################
class AlcraftWilliamsAssociation:
    def __init__(self, dfA, dfB=pd.DataFrame(data={}),method='diff',bins=10,piters=0):
        self.dfA = dfA
        self.dfB = dfB
        self.convolved = False
        if dfB.empty:
            self.convolved = True
        self.method = method
        self.associations = {}
        self.bins = bins
        self.piters = piters

    ##### Public class interface ###################################################################################    
    def addAssociation(self,cols):
        key = self.__getKey(cols)
        # https://numpy.org/doc/stable/reference/generated/numpy.histogramdd.html
        nd_data = []                
        ranges = []
        for col in cols:
            nd_data.append(self.dfA[col])
            if not self.convolved:
                amin = min(self.dfA[col].min(), self.dfB[col].min())
                amax = max(self.dfA[col].max(), self.dfB[col].max())
                range = [amin,amax]
                ranges.append(range)                                
        if self.convolved:
            histA = np.histogramdd(nd_data, bins=self.bins, density=False)[0]            
        else:            
            histA = np.histogramdd(nd_data, bins=self.bins, density=False,range=ranges)[0]
            
        histB = None
        if not self.convolved:
            nd_data = []
            for col in cols:
                nd_data.append(self.dfB[col])                        
            histB = np.histogramdd(nd_data, bins=self.bins, density=False,range=ranges)[0]            
        else:
            histB = self.__getConvolvedNDim(histA,self.dfA,cols)                
        if self.method == 'k-l':
            stat, histD = self.__calcMetric_kullbackLeibler(cols, histA, histB)
        else:
            stat,histD = self.__calcMetric_AbsDifference(cols,histA,histB)

        histA = np.transpose(histA)
        histB = np.transpose(histB)
        histD = np.transpose(histD)
        assoc = Association(cols,histA,histD,histB,stat)
        
        pval,A,B = self.calcPValue(cols)
        assoc.pvalue = pval
        assoc.phistA = A
        assoc.phistB = B
                
        self.associations[key] = assoc
        return assoc

    def calcPValue(self,cols):
        histsA = []
        histsB = []

        if self.piters > 0:            
            for i in range(self.piters):
                newA = self.getResampledData(self.dfA,cols)
                if self.convolved:
                    newB = self.getShuffledData(self.dfA,cols)
                else:
                    newB = self.getResampledData(self.dfA,cols)
                fake_aw_a = AlcraftWilliamsAssociation(newA,method=self.method,bins=self.bins)
                assoc_a = fake_aw_a.addAssociation(cols)
                fake_aw_b = AlcraftWilliamsAssociation(newB,method=self.method,bins=self.bins)
                assoc_b = fake_aw_b.addAssociation(cols)
                histsA.append(assoc_a.metric)
                histsB.append(assoc_b.metric)                
            
            # the pvalue is the area they share                                                                                    
            pmin = max(histsA)
            pmax = min(histsB)
            if max(histsA) > max(histsB):
                pmin = max(histsB)
                pmax = min(histsA)                                        
            count = len(histsA)                    
            count_between = 0                  
            for hs in histsA:
                if hs >= pmax and hs <= pmin:
                    count_between +=1                        
            for hs in histsB:
                if hs >= pmax and hs <= pmin:
                    count_between +=1                        
            total_area = 2 - (count_between/count)/2
            under_area = (count_between/count)/2            
            p_value = under_area / total_area
            return p_value,histsA,histsB
        else:
            return -1,[],[]
        
    
    def getAssociation(self,cols):
        if self.__getKey(cols) in self.associations:
            return self.associations[self.__getKey(cols)]
        else:
            self.addAssociation(cols)
            if self.__getKey(cols) in self.associations:
                return self.associations[self.__getKey(cols)]
            else:
                return None

    def getShuffledData(self,data,cols):
        dic_cut= {}
        for col in cols:
            cut_data = list(data[col].values)
            random.shuffle(cut_data)
            dic_cut[col] = cut_data
        df_cut = pd.DataFrame.from_dict(dic_cut)
        return df_cut

    def getResampledData(self,data,cols):        
        dataresampled = data.sample(frac=1,replace=True)        
        return dataresampled
    
    ##### Public and private class interface ########################################################################
    def getMetric(self,cols):
        if self.method == 'k-l':
            return self.getMetric_kullbackLeibler(cols)
        else:
            return self.getMetric_AbsDifference(cols)

    def getMetric_kullbackLeibler(self,cols):
        return 0

    def getMetric_AbsDifference(self,cols):
        return 0
    ###### Private class interface ##################################################################################
    def __getKey(self,cols):
        key = ''
        for col in cols:
            key += col + '_'
        return key

    def __getAllPermutations(self,data):
        perms = []
        for index, x in np.ndenumerate(data):
        #for index in np.ndindex((3,3,3)):
            perms.append(index)
        shaped = data.reshape(len(perms))
        return perms, shaped, data.shape

    def __setAllPermutations(self,data,shape):
        shaped = data.reshape(shape)
        return shaped

    def __getConvolvedNDim(self,orig,data,cols):
        perms, shaped, shape = self.__getAllPermutations(orig)
        hists = []
        for c in range(len(cols)):
            col = cols[c]
            histX, binsX = np.histogram(self.dfA[col], bins=self.bins, density=False)
            hists.append(histX)
        for i in range(len(shaped)):
            val = 1
            perm = perms[i]
            for p in perm:
                for c in range(len(cols)):
                    col = cols[c]
                    histX = hists[c]
                    val *= histX[p]
            shaped[i] = val
        shaped = self.__setAllPermutations(shaped, shape)
        return shaped

    def __calcMetric_kullbackLeibler(self,cols,histA,histB):
        shape = histA.shape
        vecA = histA.reshape(-1)
        vecB = histB.reshape(-1)
        vecD = np.zeros(len(vecA))
        vecA = self.__normalise(vecA)
        vecB = self.__normalise(vecB)
        stat = 0
        for i in range(len(vecA)):
            a = vecA[i]
            b = vecB[i]
            if a > 0 and b > 0: #div by and log zeros avoided
                diff = a * math.log(a/b)
                vecD[i] = diff
                stat += diff
            #print(stat,a,b)

        return stat,vecD.reshape(shape)

    def __calcMetric_AbsDifference(self,cols,histA,histB):
        shape = histA.shape
        vecA = histA.reshape(-1)
        vecB = histB.reshape(-1)
        vecD = np.zeros(len(vecA))
        vecA = self.__normalise(vecA)
        vecB = self.__normalise(vecB)
        stat = 0
        for i in range(len(vecA)):
            a = vecA[i]
            b = vecB[i]
            diff = a-b
            vecD[i] = diff
            stat += abs(diff)
            #print(a,b,diff,stat)
        # a norm step adjusts the metric for bons        
        stat = stat / 2
        #print(stat)
        #stat = stat / (1 - (1/self.bins))
        #print(stat)
        return stat,vecD.reshape(shape)

    def __normalise(self,vec):
        sum = vec.sum()
        for i in range(len(vec)):
            vec[i] = vec[i] / sum
        return vec

















