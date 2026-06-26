# try polars to accelerate the data processing
import numpy as np
from scipy import signal
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
# compute the speed slightly upstream of the bottleneck
data_upstream = data.filter((pl.col('LocalY') >= upstream_loc) & (pl.col('LocalY') <= upstream_loc + space_interval))
data_bottleneck = data.filter(
    (pl.col('LocalY') >= bottleneck_loc) & (pl.col('LocalY') <= bottleneck_loc + space_interval))
data_downstream = data.filter(
    (pl.col('LocalY') >= downstream_loc) & (pl.col('LocalY') <= downstream_loc + space_interval))
# compute the flow slightly downstream of the bottleneck
bottleneck_cd = data.filter(
    (pl.col('LocalY') >= bottleneck_loc - space_interval) & (pl.col('LocalY') <= bottleneck_loc))
# group by the FrameID column on a window of 30-second time interval and compute the average speed
avg_speed_upstream = data_upstream.group_by_dynamic("time", every="5m").agg(pl.col("MeanSpeed").mean())
avg_speed_bottleneck = data_bottleneck.group_by_dynamic("time", every="5m").agg(pl.col("MeanSpeed").mean())
avg_speed_downstream = data_downstream.group_by_dynamic("time", every="5m").agg(pl.col("MeanSpeed").mean())
total_vehicle = bottleneck_cd.group_by_dynamic("time", every="5m").agg(pl.col("VehID").n_unique())
flow_bottleneck = total_vehicle['VehID'].to_numpy() * 3600 / 60 / 5
flow_bottleneck = flow_bottleneck[1:]
density_bottleneck = flow_bottleneck / (3.6 * ft_to_m * avg_speed_bottleneck['MeanSpeed'])

# keep the hours and minutes only
t = avg_speed_bottleneck['time'].dt.strftime('%H:%M').to_numpy()
avg_speed_upstream, avg_speed_bottleneck, avg_speed_downstream = avg_speed_upstream['MeanSpeed'].to_numpy(), \
    avg_speed_bottleneck['MeanSpeed'].to_numpy(), avg_speed_downstream['MeanSpeed'].to_numpy()

# save the average speed data and date time in four columns with headers in one Excel file
df_output = pl.DataFrame(
    {'Time': t, 'Avg_speed_downstream': avg_speed_downstream, 'Avg_speed_bottleneck': avg_speed_bottleneck,
     'Avg_speed_upstream': avg_speed_upstream})
df_output.write_excel('../data/avg_speed_I24.xlsx')

# process the speed time series using wavelet transform
widths = np.arange(1, 3)
# # pywt wavelet transform
# cwtmatr, freqs = pywt.cwt(avg_speed_bottleneck, widths, 'mexh')
# cwt_bottleneck = np.mean(abs(cwtmatr) ** 2, axis=0)
# scipy wavelet transform
cwc = signal.cwt(avg_speed_bottleneck, signal.ricker, widths)
cwt_bottleneck = np.mean(abs(cwc), axis=0)

# plot the average speed and average wavelet-based energy vs. t
fig, ax = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1.5, 1]})
ax[0].plot(t, avg_speed_downstream * ft_to_m, c='k', label='MM 58.7', marker='.')
ax[0].plot(t, avg_speed_bottleneck * ft_to_m, c='b', label='MM 59.1', marker='.')
ax[0].set_ylabel('Speed (m/s)')
ax[0].legend(loc='upper left', fontsize=20)
ax[0].set_ylim([0, 40])
ax[1].plot(t, cwt_bottleneck, c='b', marker='.')
ax[1].set_ylabel('Absolute wavelet coefficient')
ax[1].set_xlabel('Time (h:mm)')
# show the xticks in the format of h:mm and select the xticks to show
ax[1].set_xticks(t[::6])
plt.xticks(rotation=45)

# # plot the bottleneck flow and density vs. time in one figure with two y-axis
# plt.figure()
# ax1 = plt.subplot(111)
# ax1.plot(t, flow_bottleneck, c='k', label='Flow', marker='o')
# ax1.set_ylabel('Flow (veh/h)')
# ax2 = ax1.twinx()
# ax2.plot(t, density_bottleneck, c='b', label='Density', marker='o')
# ax2.set_ylabel('Density (veh/km)')
# ax1.legend(loc='upper left')
# ax2.legend(loc='upper right')
# plt.figure()
# plt.plot(pqf, c='g', label='pqf', marker='o')
# plt.plot(qdf, c='k', label='qdf', marker='o')
# plt.legend()
plt.show()
