# Python Hypoxia Plume Visualization/Analysis (breastcancer-hypoxia, PhysiCell v.1.6.1)
# Anaconda distribution (https://www.anaconda.com/) recommended 
# Additional packages required in working directory: numpy, matplotlib, pyMCDS

from pyMCDS import pyMCDS
import numpy as np
import matplotlib.pyplot as plt
from numpy import linspace
import os
import scipy.cluster.hierarchy as hcluster
import math
from math import pi
 
# choose output file: '[initial, 00000xxx, final].xml'
mcds = pyMCDS('final.xml', '/home/morg/breastcancer-hypoxia/output')

# cell labels
# Set our z plane and get our substrate values along it
z_val = 0.00
 
# get our cells data and figure out which cells are in the plane
cell_df = mcds.get_cell_df()
ds = mcds.get_mesh_spacing()
inside_plane = (cell_df['position_z'] < z_val + ds) & (cell_df['position_z'] > z_val - ds)
plane_cells = cell_df[inside_plane]
non_probe = (cell_df['cytoplasmic_color_2'] == 0.0)

# Create color gradient
colors = ['black', 'blue', 'red', 'brown', 'orange', 'yellow', 'green']

# Color dead/hypoxoprobe cells and alive cells across rudimentary GFP+ gradient
sizes = [10, 15, 15, 15, 15, 15, 15]
labels = ['Dead', 'Hypoxoprobes', 'Normoxic','Perturbed', 'Medioxic','Prehypoxic', 'Hypoxic']

# set up the figure area for main tumor plot and plume analysis
fig= plt.figure(figsize=(9,4))
ax = fig.add_subplot(121)
ax2 = fig.add_subplot(122)

# start with dead/hypoxoprobe cells, and keep track of alive cells
# hypoxoprobe character: cytoplasmic_color_2
hypoxoprobe_cells = plane_cells[plane_cells['cytoplasmic_color_2']>0]
dead_cells = plane_cells[plane_cells['cycle_model'] > 100]
alive_cells = plane_cells[(plane_cells['cycle_model'] <= 100) & (plane_cells['cytoplasmic_color_2']==0)]

# pull post-hypoxic character from alive, non-probe cells
# collect live cells that are not mostly GFP+
signalled_cells = alive_cells[(alive_cells['cytoplasmic_color_1']!=0) & (alive_cells['cytoplasmic_color_0']> alive_cells['cytoplasmic_color_1'])]
# now create GFP+ thresholds
max_diff = (signalled_cells['cytoplasmic_color_0']-signalled_cells['cytoplasmic_color_1']).max()
partition = linspace(0,max_diff, 4)

# assign red/green cell types (colors) based on GFP+ thresholds
normoxic_cells = alive_cells[alive_cells['cytoplasmic_color_1']==0]
perturbed_cells = signalled_cells[(partition[2] <= (signalled_cells['cytoplasmic_color_0']-signalled_cells['cytoplasmic_color_1'])) & ((signalled_cells['cytoplasmic_color_0']-signalled_cells['cytoplasmic_color_1']) < partition[3])]
mig1_cells =signalled_cells[(partition[1] <= (signalled_cells['cytoplasmic_color_0']-signalled_cells['cytoplasmic_color_1'])) & ((signalled_cells['cytoplasmic_color_0']-signalled_cells['cytoplasmic_color_1']) < partition[2])]
mig2_cells = signalled_cells[(partition[0] <= (signalled_cells['cytoplasmic_color_0']-signalled_cells['cytoplasmic_color_1'])) & ((signalled_cells['cytoplasmic_color_0']-signalled_cells['cytoplasmic_color_1']) < partition[1])]
gfp_cells = alive_cells[alive_cells['cytoplasmic_color_1'] > alive_cells['cytoplasmic_color_0']]
s_cells = alive_cells[alive_cells['cytoplasmic_color_1']!=0]

# plot the cell layer
for i, plot_cells in enumerate((dead_cells, hypoxoprobe_cells, normoxic_cells, perturbed_cells, mig1_cells, mig2_cells, gfp_cells)):
    ax.scatter(plot_cells['position_x'].values,plot_cells['position_y'].values,facecolor='black',edgecolors=colors[i],alpha=0.6,s=sizes[i],label=labels[i])

ax.set_title("(Tp, b*, Fr) = (%d, %f, %d)" % (gfp_cells['persistence_time'].mode().max(), plane_cells['migration_bias'].max(), len(s_cells[s_cells['migration_bias']!=.1791])/len(s_cells)))

# --------------------------------------------
# plume analysis
# first evaluate proximity, then evaluate linearity

# want to ignore the perinecrotic torus of more balanced GFP+/DSRed+
# associate close perinecrotic boundary with furthest hypoxoprobe cell
hyp_x = hypoxoprobe_cells['position_x']
hyp_y = hypoxoprobe_cells['position_y']
hyp_r = np.sqrt(np.square(hyp_x) + np.square(hyp_y))
close_peri_boundary = hyp_r.max()

# associate far perinecrotic boundary with closest "perturbed" cell
pert_x = perturbed_cells['position_x']
pert_y = perturbed_cells['position_y']
pert_r = np.sqrt(np.square(pert_x) + np.square(pert_y))
far_peri_boundary = pert_r.max()

# collect all cells in plume region (past balanced perinecrotic ring)
# tweak boundary via far bias (far_bias=2 -> perinecrotic ring ends at farthest cell from origin with any GFP+)
far_bias = 1.3
close_bias = 2-far_bias
outer_ring_cells = plane_cells[np.sqrt(np.square(plane_cells['position_x'])+np.square(plane_cells['position_y']))>(far_bias*far_peri_boundary+close_bias*close_peri_boundary)/2]

# collect balanced or mostly GFP+ cells in plume region
plume_cells = outer_ring_cells

# GFP+ cells and count
green_plume_cells = outer_ring_cells[outer_ring_cells['cytoplasmic_color_0'] < outer_ring_cells['cytoplasmic_color_1']]
cell_count = green_plume_cells.shape[0]

# cluster analysis in DSRed+ layer
data = green_plume_cells.values[:, 1:3]
# set sensitivity for proximity-based cluster analysis (microns)
thresh = 75.0
clusters = hcluster.fclusterdata(data, thresh, criterion = "distance")

# plume θ range
c_count = clusters.max()
#empty array for index storage
plume_indices =np.empty(c_count, dtype = object)
for i in range(0, c_count):
      plume_indices[i]=[]

# group cells in clusters
for i in range(0, len(clusters)-1):
     plume_indices[clusters[i]-1].append(i)

plume_groups = np.empty(c_count, dtype = object)
for i in range(0, c_count):
   plume_groups[i] = green_plume_cells.iloc[plume_indices[i]]

# initialize plume data columns
theta_vals = np.empty(c_count, dtype = object)
theta_deg_diff = np.empty(c_count, dtype = object)
r_vals =np.empty(c_count, dtype = object)
max_depth =np.empty(c_count, dtype = object)

# collect breadth data
for i in range(0, c_count):
    if (len(plume_groups[i])>0):
      theta_vals[i] = np.arctan2(plume_groups[i]['position_y'], plume_groups[i]['position_x'])
      if (theta_vals[i].max()-theta_vals[i].min() >pi):
            theta_vals[i] = np.arctan2(plume_groups[i]['position_y'], (-1)*plume_groups[i]['position_x'])
      theta_deg_diff[i] = (theta_vals[i].max()-theta_vals[i].min())*360/(2*pi)
    else:
      theta_vals[i]=0
      theta_deg_diff[i]=0

# compute typical plume breadth as a weighted average
avg_breadth = 0
for i in range(0, c_count):
      avg_breadth+=(theta_deg_diff[i]*len(plume_groups[i]['ID'])/len(clusters))

# collect depth data
for i in range(0, c_count):
     if (len(plume_groups[i])>0):
       r_vals[i] = np.sqrt(np.square(plume_groups[i]['position_x']) + np.square(plume_groups[i]['position_y']))
       max_depth[i] = r_vals[i].max()
     else:
       r_vals[i]=0
       max_depth[i]=0

# compute typical plume depth as a weighted average
avg_depth = 0
for i in range(0, c_count):
      avg_depth+=(max_depth[i]*len(plume_groups[i]['ID'])/len(clusters))

# write in data
with open('plume_data.txt', 'a') as f:
	f.write('%d %f %f %f %f %d\n' % (green_plume_cells['persistence_time'].mode().max(), plane_cells['migration_bias'].max(), len(s_cells[s_cells['migration_bias']!=.1791])/len(s_cells), avg_breadth, avg_depth, c_count))
f.close()

# cluster analysis plotting
ax2.scatter(*np.transpose(data), c=clusters)
plt.axis("equal")
title = "Number of plumes: %d\nBreadth/Depth: (%f, %f)" % (len(set(clusters)), avg_breadth, avg_depth)
ax2.set_title(title)
plt.show()
