import helpers as hlp
import ra_trivial.trivial as awa
import pandas as pd

# CELL 1 ############################################


#columns =['A', 'B','C','AB','BC','CA','CBp']
#columns =['A', 'B','C']
columns =['A', 'B','C','BC']
bins = 6
no_tri= 12
max_dims = len(columns)
#dic_lists = {}

# CELL 2 ############################################
# generate a triangle dataset
dfR = hlp.make_triangles(no_tri,"R",columns)
dfA = hlp.make_triangles(no_tri,"A",columns)
dfM = hlp.make_triangles(no_tri,"M",columns)
print(dfM)

dic_dfs_per_tri = {}
dic_dfs_per_tri["Mixed"] = dfM
dic_dfs_per_tri["Right"] = dfR
dic_dfs_per_tri["Any"] = dfA

print("Completed")
# CELL 3 ############################################
# Create the trivial divergence
cols = columns

sets_corrs = {}

for tag,df in dic_dfs_per_tri.items():
  rae_mark = awa.AlcraftWilliamsAssociation(df,bins=bins,piters=0)
  for i in range(2,max_dims+1):
    print(tag,"\tmake",i,"dims")
    df_corri = rae_mark.getStrongestAssociations([],cols,i,1,sort=False)
    sets_corrs[f"{tag}_{i}"] = df_corri
    
print("Complete")

# CELL 4 ##############################################
hlp.heatmap_by_removal(dic_dfs_per_tri,max_dims,sets_corrs,cols,bins)

# CELL 5 #############################################
# Split into all possible halves and retry
# First the original data
#for tag in ["Mixed","Right","Any"]:
for tag in ["Mixed"]:
    df_obs = dic_dfs_per_tri[tag]
    dic_all_metrics = {}
    print("Calc all")
    use_diff = False
    dic_all_metrics = hlp.calc_and_add_triv(df_obs,dic_all_metrics,bins,max_dims,cols,use_diff=use_diff)
    
    # Create the samples
    fraction = 3
    obs_list = range(0,len(df_obs.index))
    sample_obs = int(len(df_obs.index)/fraction)
    #sample_obs = len(df_obs.index) - 1
    print("There are",len(df_obs.index),"observations","half=",sample_obs)
    hlp.heatmap_by_sample(obs_list, sample_obs,bins,max_dims,cols,df_obs,dic_all_metrics,tag)
    
    # make into states
    print("create states")

