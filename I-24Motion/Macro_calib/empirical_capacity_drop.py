# try polars to accelerate the data processing
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import polars as pl
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# globally set font size and style for xylabel and legend and text
plt.rcParams.update({'font.size': 30, 'font.family': 'Times New Roman'})
plt.rcParams["mathtext.fontset"] = "cm"
# # "FrameID", "LaneID", "VehID", "VehClass", "Vehlen", "LocalY", "MeanSpeed"
ft_to_m = 0.3048
# unit: ft/s
data = pl.read_parquet("../data/I24_data_lane2.parquet")
# sort the data by FrameID
data = data.sort(by=['FrameID'])
# convert the FrameID in Unix timestamps to datetime in timezone UTC-6 CST
data = data.with_columns(((pl.from_epoch(pl.col('FrameID'))).dt.offset_by('-6h')).alias('time'))
# set_sorted the column time
data = data.set_sorted(column=['time'])
# change the UTC timezone to CST timezone
# compute the average speed in every 30 seconds at upstream and downstream of the bottleneck
upstream_loc, bottleneck_loc, downstream_loc = 59.2 * 5280, 59.1 * 5280, 58.7 * 5280  # location for I24 (ft)
space_interval = 100  # ft
# compute the flow slightly downstream of the bottleneck
bottleneck_cd = data.filter(
    (pl.col('LocalY') >= bottleneck_loc - space_interval) & (pl.col('LocalY') <= bottleneck_loc))
total_vehicle = bottleneck_cd.group_by_dynamic("time", every="5m").agg(pl.col("VehID").n_unique())
# congestion lasts from 6:55 to 8:50, while free flow is from 9:10 to 10:00
# the capacity is the 5-min maximum flow during the free flow period
free_flow = total_vehicle.filter(pl.col('time').is_between(datetime(2022, 11, 22, 9, 10),
                                                           datetime(2022, 11, 22, 10, 0)))
capacity = np.max(free_flow['VehID'].to_numpy()) * 3600 / 60 / 5
print("The capacity is: ", capacity)
# the queue discharge rate is the average flow during the congestion period
congestion = total_vehicle.filter(pl.col('time').is_between(datetime(2022, 11, 22, 6, 55),
                                                            datetime(2022, 11, 22, 8, 50)))
queue_discharge_rate = np.mean(congestion['VehID'].to_numpy()) * 3600 / 60 / 5
print("The queue discharge rate is: ", queue_discharge_rate)
# compute the capacity drop
capacity_drop = (capacity - queue_discharge_rate) / capacity
print("The capacity drop is: ", capacity_drop * 100, "%")
# plot the flow time series
# sort the data by time
congestion = congestion.sort(by=['time'])
free_flow = free_flow.sort(by=['time'])
fig, ax = plt.subplots(1)
# green for free flow, red for congestion
ax.plot(congestion['time'].dt.strftime('%H:%M').to_numpy(), congestion['VehID'].to_numpy() * 3600 / 60 / 5, 'r',
        label='Congestion')
ax.plot(free_flow['time'].dt.strftime('%H:%M').to_numpy(), free_flow['VehID'].to_numpy() * 3600 / 60 / 5, 'g',
        label='Free flow')

plt.xticks(rotation=45)
plt.xlabel('Time (h:mm)')
plt.ylabel('Flow (veh/h)')
plt.legend()
plt.show()
# The capacity is:  1776.0
# The queue discharge rate is:  1465.0
# The capacity drop is:  17.51126126126126 %
