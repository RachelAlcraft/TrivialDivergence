'''
This test is to check if the seach for multi dim association to a col works
'''
import os
import sys
import numpy as np
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + '/../')
import ra_trivial.datasets as ds
import ra_trivial.trivial as awa
import ra_trivial.reports as re



# Generate random and unknonw dataset
vertices = [3,4]
samples = 3
grid_size = 10
shapes_set = ds.DataSetShapes(vertices,samples=samples,grid_size=grid_size)
anon_df = shapes_set.getAnonDataFrame()
id_df = shapes_set.getIdentifiableDataFrame()
print(anon_df)
print(id_df)
print(shapes_set.shapes_frame)

# we now go through all the associations in 2d, 3d and 4d
# we will then print out the results
bins=5
piters=5
method='diff'
# I want to compare the data against itself for clustering, so I need to repat the columns
anon_df2 = anon_df.copy()
anon_df3 = anon_df.copy()
anon_df4 = anon_df.copy()
anon_df2['angle2'] = anon_df['angle']
anon_df3['angle2'] = anon_df['angle']
anon_df3['angle3'] = anon_df['angle']
anon_df4['angle2'] = anon_df['angle']
anon_df4['angle3'] = anon_df['angle']
anon_df4['angle4'] = anon_df['angle']
# create the association object
# Associations in 2d
rae_mark2 = awa.AlcraftWilliamsAssociation(anon_df2,bins=bins,piters=piters,method=method,loglevel=2)
df2d = rae_mark2.getStrongestAssociations(['angle'],['angle2'],1,1)
print(df2d)
# Associations in 3d
rae_mark3 = awa.AlcraftWilliamsAssociation(anon_df3,bins=bins,piters=piters,method=method,loglevel=2)
df3d = rae_mark3.getStrongestAssociations(['angle'],['angle2','angle3'],2,1)
print(df3d)
# Associations in 4d
rae_mark4 = awa.AlcraftWilliamsAssociation(anon_df4,bins=bins,piters=piters,method=method,loglevel=2)
df4d = rae_mark4.getStrongestAssociations(['angle'],anon_df4.columns,2,1)
print(df4d)




# Print out the orig data, but order it so that we can compare it visually
id_df.sort_values(by=['angle'],inplace=True)
id_df.index = range(len(id_df))
id_df['idx'] = id_df.index
fig = make_subplots(rows=3, cols=1,horizontal_spacing=0.05,vertical_spacing=0.05,column_widths=[1])


fig1 = px.scatter(id_df,x='angle',y='idx',color="shape",title='Identifiable Data',width=800,height=400)
fig2 = px.line(shapes_set.shapes_frame, x="x", y="y", color="shapeid", text="shape")

fig.add_trace(fig1.data[0],row=1,col=1)
for i in range(len(fig2.data)):
    fig.add_trace(fig2.data[i],row=2,col=1)





fig.write_html(os.path.dirname(os.path.realpath(__file__)) + '/output/Tests08.html')

