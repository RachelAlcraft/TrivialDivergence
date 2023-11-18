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
    def __init__(self, dfA, dfB=pd.DataFrame(data={}),method='diff',bins=10,piters=0,loglevel=0):
        self.dfA = dfA
        self.dfB = dfB
        self.convolved = False
        if dfB.empty:
            self.convolved = True
        self.method = method
        self.associations = {}
        self.bins = bins
        self.piters = piters
        self.loglevel=loglevel

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
        histB = None
        if self.convolved:
            histA = np.histogramdd(nd_data, bins=self.bins, density=False)[0]            
            histB = self.__getConvolvedNDim(histA,cols)                
        else:            
            histA = np.histogramdd(nd_data, bins=self.bins, density=False,range=ranges)[0]
            nd_data = []
            for col in cols:
                nd_data.append(self.dfB[col])                        
            histB = np.histogramdd(nd_data, bins=self.bins, density=False,range=ranges)[0]            
                                        
        if self.method == 'k-l':
            stat, histD = self.__calcMetric_kullbackLeibler(cols, histA, histB)
        else:
            stat,histD = self.__calcMetric_AbsDifference(cols,histA,histB)

        histA = np.transpose(histA)
        histB = np.transpose(histB)
        histD = np.transpose(histD)
        assoc = Association(cols,histA,histD,histB,stat)
        
        if self.convolved:
            pval,A,B = self.calcPValueAA(cols)
        else:
            pval,A,B = self.calcPValueAB(cols)
        assoc.pvalue = pval
        assoc.phistA = A
        assoc.phistB = B
                
        self.associations[key] = assoc
        return assoc

    def calcPValueAA(self,cols):
        '''
        The null hypothesis is that the distribution is entirely random
        We shuffle it
        And we shuffle random
        And plot 2 histograms
        Where they overlap is the p-value
        '''
        histsA = []
        histsB = []

        if self.piters > 0:            
            for i in range(self.piters):
                newA = self.getResampledData(self.dfA,cols)                
                newB = self.getShuffledData(self.dfA,cols)                
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
        
    def calcPValueAB(self,cols):
        '''
        The null hypothesis is that The A distribution is drawn from the B distribution
        We find the metric that is the difference, then we see how likely it is that B could have come up with that metric
        byt sampling and recalulating the difference with itself 
        And resampling and recalculating theA with it
        And seeing where they overlap
        '''        
        histsA = []
        histsB = []

        if self.piters > 0:            
            for i in range(self.piters):
                newA = self.getResampledData(self.dfA,cols)                
                newB = self.getResampledData(self.dfB,cols)                
                fake_aw_a = AlcraftWilliamsAssociation(newA,self.dfB,method=self.method,bins=self.bins)
                assoc_a = fake_aw_a.addAssociation(cols)
                fake_aw_b = AlcraftWilliamsAssociation(newB,self.dfB,method=self.method,bins=self.bins)
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
    
    def getStrongestAssociations(self,colsA,colsB,dims_plus,fraction=1.0,sort=True):
        cols_sure = list(colsA)
        cols_in_scope = list(colsB)
        for col in cols_sure:            
            if col in cols_in_scope:
                cols_in_scope.remove(col)            
        # first we are only going to have in scope those that have a reasonable stat in 2d. The fraction is used.
        if fraction < 1 and dims_plus > 1 and len(colsA) == 1: #it only makes to try the cols that are 2d associated to 1
            df2 = self.getStrongestAssociations_inner(cols_sure,cols_in_scope,1,sort=sort)
            last_col = len(cols_sure)+1 #it will be the first col
            rows = int(len(df2.index)*fraction)
            if len(colsA)>0:
                df2 = df2.head(rows)
            cols_cut = df2['col' + str(last_col)].values                
        else:
            cols_cut = cols_in_scope
        dfall = self.getStrongestAssociations_inner(cols_sure,cols_cut,dims_plus,sort=sort)
        dfall = dfall.reset_index()
        return dfall

    def getStrongestAssociations_inner(self,colsA,colsB,dims_plus,sort=True):
        if self.loglevel>1:
            print('AWDiv(2) strongest inner:',colsA,len(colsB))
        import itertools
        dims_all = len(colsA)+dims_plus
        combs = list(itertools.combinations(colsB,dims_plus))                               
        if self.loglevel>1:
            print('AWDiv(2) len combos:',len(combs))
        strongest_dic = {'metric':[]}
        for c in range(dims_all):
            strongest_dic['col'+str(c+1)] = []                        
        
        count = 0
        for comb in combs:
            if self.loglevel>1:
                if count%100==0:
                    print('AWDiv(2)',count,'/',len(combs),comb)
            count+=1
            col_list = []
            for c in range(len(colsA)):                        
                strongest_dic['col'+str(c+1)].append(colsA[c])
                col_list.append(colsA[c])
            c = len(colsA)+1
            for cb in comb:
                col_list.append(cb)
                strongest_dic['col' + str(c)].append(cb)
                c+=1                        
            asso = self.getAssociation(col_list)
            strongest_dic['metric'].append(asso.metric)                
        df_strongest = pd.DataFrame.from_dict(strongest_dic)
        if sort:
            df_strongest = df_strongest.sort_values(by='metric',ascending=False)
        return df_strongest
            
        
    
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

    def __getConvolvedNDim(self,orig,cols):
        perms, shaped, shape = self.__getAllPermutations(orig)
        hists = []
        for c in range(len(cols)):
            col = cols[c]
            histX, binsX = np.histogram(self.dfA[col], bins=self.bins, density=False)
            hists.append(histX)
        for i in range(len(shaped)):
            val = 1
            perm = perms[i]
            for j in range(len(perm)):
                #for c in range(len(cols)):
                #    col = cols[c]
                #    histX = hists[c]
                #    val *= histX[p]                
                p = int(perm[j])
                histX = hists[j]
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

















