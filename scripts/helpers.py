import pandas as pd
import random
import math
import plotly.express as px
import ra_trivial.trivial as awa
import itertools

# triangle making function
def make_triangles(no_tri,right_all_mix,columns):

  # R=Right, A=any, M e half right half any for 2 states
  bond_length_angle_range = (45,55)
  bond_length_range_lower = (5,15)
  bond_length_range_mid = (5,15)
  bond_length_range_upper = (5,15)
  AB_range = [5,8]
  dic_lists = {}

  As = []
  Bs = []
  Cs = []
  ABs = []
  BCs = []
  CAs = []
  CBps = []

  for i in range(no_tri):
    this_r_a_m = right_all_mix
    if right_all_mix == "M":
      if i < no_tri/2:
        this_r_a_m = "R"
      else:
        this_r_a_m = "A"
    if this_r_a_m == "R":
      A = 90
    elif i == 0:
      A = 90 #I want to make sure I capture the random possibility of a different state so always have at least 1 right angle triangle
    else:
      A = random.randint(1, 178)
    B = random.randint(1, 180-A-1)
    C = 180 - (A+B)
    Ar = math.radians(A)
    Br = math.radians(B)
    Cr = math.radians(C)
    AB = random.randint(5,8)
    sin_ratio = AB / math.sin(Cr)
    BC = math.sin(Ar) * sin_ratio
    CA = math.sin(Br) * sin_ratio
    #CBp is the link to the next triangle
    if C < bond_length_angle_range[0]:
      CBp = random.randint(bond_length_range_lower[0],bond_length_range_lower[1])
    elif C > bond_length_angle_range[1]:
      CBp = random.randint(bond_length_range_upper[0],bond_length_range_upper[1])
    else:
      CBp = random.randint(bond_length_range_mid[0],bond_length_range_mid[1])
    #print(A,B,C,A+B+C,AB,BC,CA,CBp)
    As.append(A)
    Bs.append(B)
    Cs.append(C)
    ABs.append(AB)
    BCs.append(BC)
    CAs.append(CA)
    CBps.append(CBp)
  ####################################

  dic_lists["A"] = As
  dic_lists["B"] = Bs
  dic_lists["C"] = Cs
  dic_lists["AB"] = ABs
  dic_lists["BC"] = BCs
  dic_lists["CA"] = CAs
  dic_lists["CBp"] = CBps

  df_list = []
  for col in columns:
    df_list.append( dic_lists[col])

  df_transpose =  [list(i) for i in zip(*df_list)]
  df = pd.DataFrame(df_transpose,columns =columns)
  return df

#########################################################################
#########################################################################
#########################################################################
# Sampling each observation FUNCTION
# We now want to try again removing each obs and recalcing
def drop_row(df,cols,bins,dim,df_corr,show=False,only=False):
  list_x = df_corr["metric"].tolist()
  col_names = []
  for i in range(len(df_corr.index)):
    cols_nm = ""
    for c in range(dim):
      col_nm = f"col{c+1}"
      cols_nm += ":" + df_corr[col_nm][i]
    col_names.append(cols_nm[1:])
  if show:
    print(col_names)

  if show:
    print("x",list_x)
  # list of lists object for heatmap
  list_of_lists = []
  list_of_metrics = []
  for i in range(len(df.index)):
    dfx=df.drop(df.index[i])
    #if only: #instead of dropping we do it on its own
    #  dfx=df.index[i]
    rae_markx = awa.AlcraftWilliamsAssociation(dfx,bins=bins,piters=0)
    df_corrx = rae_markx.getStrongestAssociations([],cols,dim,1,sort=False)
    list_i = df_corrx["metric"].tolist()    
    df_corrx['is_it_bigger'] = df_corrx['metric'] > (df_corr['metric'])
    #df_corrx["is_it_bigger"] = df_corrx["is_it_bigger"].astype(str) #make it a string
    list_TF = df_corrx["is_it_bigger"].tolist()
    list_met = df_corrx["metric"].tolist()
    #print(i,list_i)
    if show:
      print(i,list_TF)
    list_of_lists.append(list_TF)
    list_of_metrics.append(list_met)
  return col_names,list_of_lists,list_of_metrics

#########################################################################
#########################################################################
#########################################################################

  # make into a heatmap
# https://plotly.com/python/heatmaps/

def make_heatmap(list_of_lists, col_names, title,hue,save_path):

  fig = px.imshow(list_of_lists,
                  labels=dict(x="Association", y="Observations?", color=hue),
                  x=col_names,
                  color_continuous_scale='Bluered_r',                  
                  title=title,
                  range_color=[0,1])
  fig.update_xaxes(side="top")
  fig.write_html(save_path)


#########################################################################
#########################################################################
#########################################################################
def heatmap_by_removal(dic_dfs_per_tri,max_dims,sets_corrs,cols,bins):
    for tag,df in dic_dfs_per_tri.items():
        print(tag)
        show_rows = False  
        #only=True
        all_col_names = []  
        dic_lists = {}
        dic_metrics = {}
        print("Drop rows")
        for i in range(2,max_dims+1):
            print(f"{i}/{max_dims}")
            dfi = sets_corrs[f"{tag}_{i}"]    
            col_namesi,list_of_listsi,list_of_metricsi = drop_row(df,cols,bins,i,dfi,show=show_rows)        
            dic_lists[i] = list_of_listsi    
            dic_metrics[i] = list_of_metricsi
            all_col_names += col_namesi
        
        list_of_list_of_lists = []
        list_of_list_of_metrics = []
        num_obs = len(dic_lists[2]) # get the length of the first one, it is the number of observations
        print("Create obs lists")
        for ob in range(num_obs): 
            lolol = []
            for key,lol in dic_lists.items():    
                obs_lst = lol[ob]      
                lolol += obs_lst
                list_of_list_of_lists.append(lolol)
        for ob in range(num_obs): 
            lolol = []
            for key,lol in dic_metrics.items():      
                obs_lst = lol[ob]      
                lolol += obs_lst
                list_of_list_of_metrics.append(lolol)
        # Make HEATMAP
        print("number obs=",num_obs,tag)
        #make_heatmap(list_of_list_of_lists, all_col_names, title=f"{tag} Improves by removal?",hue="Improves?")
        make_heatmap(list_of_list_of_metrics, all_col_names,title=f"{tag} Metric with removal",hue="Triviality",save_path=f"plots/{tag}_removal.html")

#########################################################################
#########################################################################
#########################################################################
# Function to calculate association and add it to a dictopnary
def calc_and_add_triv(df_obs,dic_all_metrics,bins,max_dims,cols,use_diff=False):
  #df_obs = dic_dfs_per_tri["Mixed"]
  rae_mark_obs = awa.AlcraftWilliamsAssociation(df_obs,bins=bins,piters=0)
  dic_obs = {}
  #dic_all_metrics = {}
  for i in range(2,max_dims+1):
    #print("make",i,"dims")
    df_corri = rae_mark_obs.getStrongestAssociations([],cols,i,1,sort=False)
    for idx in df_corri.index:
      triviality = df_corri["metric"][idx]
      triv_nm = ""
      for c in range(i):
        triv_nm += ":" + df_corri[f"col{c+1}"][idx]
      triv_nm = triv_nm[1:]    
      if triv_nm not in dic_all_metrics:
        dic_all_metrics[triv_nm] = [triviality]
      else:
        if use_diff:
            triv_diff = triviality - dic_all_metrics[triv_nm][0] 
            dic_all_metrics[triv_nm].append(triv_diff)
        else:
            dic_all_metrics[triv_nm].append(triviality)
      #if len(dic_all_metrics[triv_nm]) > 20:
      #  print(triv_nm)
  return dic_all_metrics  

#########################################################################
#########################################################################
#########################################################################
def lol_cols_idx_to_dataframe(lol,idxs=None,cols=None,transpose=False):
    df = pd.DataFrame(lol)    
    if transpose:
        df = df.T
    if cols is not None:        
        df.columns = cols
    if idxs is not None:
        df.index = idxs
    return df



#########################################################################
#########################################################################
#########################################################################

def heatmap_by_sample(obs_list, sample_obs,bins,max_dims,cols,df_obs,dic_all_metrics,tag,sorted=False):
    # Generate all possible two-element combinations
    # Convert the resulting iterator to a list
    combinations = list(itertools.combinations(obs_list, sample_obs))
    print("There are", len(combinations))
    for i in range(len(combinations)):
        if i%500 == 0 or i == len(combinations)-1:
            print("Calc",i,"/",len(combinations)-1)
        combo = combinations[i]
        combo_lst = []
        for cm in combo:
            combo_lst.append(cm)
        halved_df = df_obs.iloc[combo_lst]  
        dic_all_metrics = calc_and_add_triv(halved_df,dic_all_metrics,bins,max_dims,cols)
    
    # make a heatmap frtom the samples
    # turn dictonary into a list of cal names and a list of list
    lol={}
    col_header = []

    Y = []
    for triv,vals in dic_all_metrics.items():
        if Y == []:
            Y = vals[1:]        
        if sorted:
            X = vals[1:]
            vals = [x for _,x in sorted(zip(Y,X))]          
        else:
            vals = vals[1:]
        col_header.append(triv)
        for v in range(len(vals)):    
            if v not in lol:
                lol[v] = []
            lol[v].append(vals[v])
            #print(triv,v,vals[v])    
    
    lolol = []
    vols = lol[0]
    for v,vals in lol.items():  
        lolol.append(vals)
        #print(vals)
    print(col_header)
    make_heatmap(lolol, col_header,title=f"{tag} Triviality in samples of {sample_obs}",hue="Triviality Divergence",save_path=f"plots/{tag}_samples_{sample_obs}.html")
    df = lol_cols_idx_to_dataframe(lolol,cols=col_header,idxs=combinations,transpose=False)
    df["states"] = ""
    for col in col_header:
        max_col = df[col].max()
        min_col = df[col].min()        
        for idx in combinations:
            idx_val = df[col][idx]
            if idx_val == min_col:
                df["states"][idx] += "0"
            elif idx_val == max_col:
                df["states"][idx] += "2"
            else:
                df["states"][idx] += "1"
                        
    print(df)






