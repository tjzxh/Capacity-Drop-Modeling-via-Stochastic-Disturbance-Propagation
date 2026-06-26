import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import polars as pl

# globally set font size and style for xylabel and legend and text
plt.rcParams.update({'font.size': 30, 'font.family': 'Times New Roman'})
plt.rcParams["mathtext.fontset"] = "cm"
# # "FrameID", "LaneID", "VehID", "VehClass", "Vehlen", "LocalY", "MeanSpeed"
# unit: m
data = pl.read_parquet("../data/Zen_data.parquet")
# sort the data by FrameID
data = data.sort(by=['FrameID'])
# set_sorted the column time
data = data.set_sorted(column=['time'])
bottleneck_loc = 4000  # location for Zen (m)
space_interval = 100  # m
time_interval = 60  # s
# compute the flow slightly downstream of the bottleneck
data_bn = data.filter(
    (pl.col('LocalY') >= bottleneck_loc - space_interval) & (pl.col('LocalY') <= bottleneck_loc))
# compute the fundamental diagram using Edie's definitions
area = space_interval * time_interval
# density is the sum of (max_time - min_time of each vehicle)/area of the cell
# collect the maximum  and minimum frame of each vehicle
data_edie = data_bn.group_by_dynamic('time', every=str(time_interval) + 's', by='VehID').agg(
    total_time_for_each_vehicle=(pl.col('Epoch').max() - pl.col('Epoch').min()) / 1000,
    total_localy_for_each_vehicle=pl.col('LocalY').max() - pl.col('LocalY').min())
all_total_time_for_each_vehicle = data_edie['total_time_for_each_vehicle'].to_numpy()
# sum the total time and total localy for each vehicle in each cell simultaneously
fd_cell = data_edie.group_by('time').agg(time_cell=pl.col('total_time_for_each_vehicle').sum(),
                                         localy_cell=pl.col('total_localy_for_each_vehicle').sum())
# sort the data by time
fd_cell = fd_cell.sort(by=['time'])
# compute the density and flow with unit veh/km and veh/h
fd_cell = fd_cell.with_columns(density=1000 * pl.col('time_cell') / area,
                               flow=3600 * pl.col('localy_cell') / area)
# aggregate the data by time further
qk_data = fd_cell.group_by_dynamic('time', every='5m').agg(k=pl.col('density').mean(), q=pl.col('flow').mean())
qk_data = qk_data.sort(by=['time'])
# compute the density and flow with unit veh/km and veh/h
density = fd_cell['density'].to_numpy()
flow = fd_cell['flow'].to_numpy()
speed = flow / density
t = fd_cell['time'].dt.strftime('%H:%M:%S').to_numpy()
# save the data into Excel with headers: q (veh/h), k (veh/km), v (km/h)
fd_data = np.vstack((t, flow, density, speed)).T
df = pd.DataFrame(fd_data, columns=['t', 'q', 'k', 'v'])
df.to_excel('../data/fd_data.xlsx')
# plot the q-k diagram
plt.scatter(density, flow, c='b')
# set box aspect to be equal
plt.gca().set_box_aspect(1)
# plot capacity point (1920 veh/h) and queue discharge rate point (1507.2 veh/h) at critical density (k value at 10:25 of qk_data)
# kc = qk_data.filter(pl.col('time') == datetime(2022, 11, 22, 6, 10))['k'].to_numpy()
# plt.plot([35, 35], [1507, 1920], 'r', marker='o', markersize=10, linewidth=5)
plt.xlabel('Density (veh/km)')
plt.ylabel('Flow (veh/h)')
# plot the q-t, k-t diagram
agg_flow = qk_data['q'].to_numpy()
agg_density = qk_data['k'].to_numpy()
agg_time = qk_data['time'].dt.strftime('%H:%M').to_numpy()
fig, ax = plt.subplots(2, 1, sharex=True)
# plot the q-t diagram
ax[0].scatter(agg_time, agg_flow, c='b')
ax[0].set_ylabel('Flow (veh/h)')
# plot the k-t diagram
ax[1].scatter(agg_time, agg_density, c='b')
ax[1].set_xlabel('Time (h:mm)')
ax[1].set_ylabel('Density (veh/km)')
# choose proper numbers of the xticks for the q-t and k-t diagram, from 6:00 to 10:00, every 30 minutes
ax[1].set_xticks(agg_time[::3])
ax[1].tick_params(rotation=45)
plt.show()
