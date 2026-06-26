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
# unit: m
data = pl.read_parquet("../data/Zen_data.parquet")
# sort the data by FrameID
data = data.sort(by=['FrameID'])
# convert the FrameID to string
data = data.with_columns(string_frame=pl.col('FrameID').cast(pl.Utf8))
# convert string "HHMMSSFFF" to datetime
data = data.with_columns(time=pl.col('string_frame').str.to_datetime("%H%M%S%3f"))
# set_sorted the column time
data = data.set_sorted(column=['time'])
bottleneck_loc, downstream_loc = 4000, 3000  # location for Zen (m)
space_interval = 100  # m
# compute the flow slightly downstream of the bottleneck
bottleneck_cd = data.filter(
    (pl.col('LocalY') >= bottleneck_loc - space_interval) & (pl.col('LocalY') <= bottleneck_loc))
total_vehicle = bottleneck_cd.group_by_dynamic("time", every="5m").agg(ncount=pl.col("VehID").n_unique())
# congestion lasts from 10:35 to 11:00, while free flow is from 10:00 to 10:25
# the capacity is the 5-min maximum flow during the free flow period
free_flow = total_vehicle.filter(pl.col('time').is_between(datetime(1, 1, 1, 10, 5),
                                                           datetime(1, 1, 1, 10, 25)))
capacity = np.max(free_flow['ncount'].to_numpy()) * 60 / 5
print("The capacity is: ", capacity)
# the queue discharge rate is the average flow during the congestion period
congestion = total_vehicle.filter(pl.col('time').is_between(datetime(1, 1, 1, 10, 35),
                                                            datetime(1, 1, 1, 10, 59)))
queue_discharge_rate = np.mean(congestion['ncount'].to_numpy()) * 60 / 5
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
ax.plot(free_flow['time'].dt.strftime('%H:%M').to_numpy(), free_flow['ncount'].to_numpy() * 60 / 5, 'g',
        label='Free flow')
ax.plot(congestion['time'].dt.strftime('%H:%M').to_numpy(), congestion['ncount'].to_numpy() * 60 / 5, 'r',
        label='Congestion')
plt.xticks(rotation=45)
plt.xlabel('Time (h:mm)')
plt.ylabel('Flow (veh/h)')
plt.legend()
plt.show()
# The capacity is:  1920.0
# The queue discharge rate is:  1507.2
# The capacity drop is:  21.499999999999996 %
