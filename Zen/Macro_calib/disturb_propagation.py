import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import linear_model
from funcs_for_correlation import wave_propagation_polars, collect_avg_speed_polars
import polars as pl
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams.update({'font.size': 30, 'font.family': 'Times New Roman'})
plt.rcParams["mathtext.fontset"] = "cm"
# load the data and the correlation length
# unit: m, m/s
# # "FrameID", "LaneID", "VehID", "VehClass", "Vehlen", "LocalY", "MeanSpeed"
data = pl.read_parquet("../data/Zen_data.parquet")
# sort the data by FrameID
data = data.sort(by=['FrameID'])
data = data.set_sorted(column=['time'])
# filter the data that are in the isolated bottleneck (LocalY in [KP 3.65, KP 4.2]*1000 m)
data = data.filter((pl.col('LocalY') >= 3650) & (pl.col('LocalY') <= 4200))
data = data.unique(subset=['FrameID', 'VehID'])
data = data.sort(by=['LocalY'])
# read the fundamental diagram file: columns=[t, q, k, v]
fd_data = pl.read_excel('../data/fd_data.xlsx')
# read the correlation file: columns=['time', 'corr_veh', 'corr_value', 'max_veh_len', 'corr_len']
corr_data = pl.read_excel('../data/correlation_length.xlsx')
# convert the string "HH:MM:SS" to datetime
corr_data = corr_data.with_columns(date_time=pl.col('time').str.to_datetime("%H:%M:%S"))
# convert the datetime to epoch time
corr_data = corr_data.with_columns(epoch_time=pl.col('date_time').dt.epoch(time_unit='ms'))
# find the data with maximum correlation length
# calculate the critical density with maximum correlation length
idc = corr_data['corr_len'].arg_max()
critical_time = corr_data[idc]['epoch_time'][0]
print(f"critical time={critical_time} ms")
# group the data by time and get the idc th row
data = data.sort(by=['time'])
df = data.group_by_dynamic('time', every='1m').agg(epochlist=pl.col('Epoch'), vehlist=pl.col('VehID'),
                                                   speedlist=pl.col('MeanSpeed'), localylist=pl.col('LocalY'))
dfc = df[idc]
# explode the list columns of dfc
dfc = dfc.explode(['epochlist', 'vehlist', 'speedlist', 'localylist'])
avg_speed = dfc['speedlist'].mean()
# rename the columns to frame, VehID, Speed_m, LocalY_m
dfc = dfc.rename({'epochlist': 'Epoch', 'vehlist': 'VehID', 'speedlist': 'Speed', 'localylist': 'LocalY'})
# filter the data in the first frame
dfc0 = dfc.filter(pl.col('Epoch') == pl.col('Epoch').min())
print(f"the minimum frame={dfc0['Epoch'].to_numpy()[0]}")
dfc0 = dfc0.rename({'Epoch': 'tc'})
# sort the data by frame and local y
dfc0 = dfc0.sort(by=['LocalY'])
# # plot the speed vs. t for each vehicle
# all_vehicle = dfc0['VehID'].to_numpy()
# for veh in all_vehicle:
#     plt.plot(data.filter(pl.col('VehID') == veh)['Epoch'] / 1000, data.filter(pl.col('VehID') == veh)['MeanSpeed'])
# plt.xlabel('Time (s)')
# plt.ylabel('Speed (m/s)')
# plt.show()

# get the average speed of each vehicle in the fisrt frame of the critical time window
init_speed = collect_avg_speed_polars(data, dfc0)
# calculate the speed deviation of the first 3 slow-speed vehicles
slow_ind = np.where(init_speed < avg_speed)[0]
init_disturbance = np.mean(init_speed[slow_ind[0]]) - avg_speed
print(f"initial disturbance={init_disturbance} m/s")
# initial disturbance=-0.7580311736263425 m/s
# # plot the initial disturbance
# plt.figure()
# plt.plot(dfc0['LocalY'], init_speed, marker='o', c='b', lw=1)
# plt.plot([dfc0['LocalY'].min(), dfc0['LocalY'].max()], [avg_speed, avg_speed], c='k', lw=1)
# plt.xlabel('Local Y (m)')
# plt.ylabel('Speed (m/s)')
# plt.show()

# get the data around the critical time (+-10 s)
dfc = data.filter((pl.col('Epoch') >= critical_time) & (pl.col('Epoch') <= critical_time + 60 * 1000))
avg_speed = dfc['MeanSpeed'].mean()
follower_dev, leader_dev, possible_init_disturb = [], [], []
# sample from the low-speed vehicles
sample_data = dfc.filter(pl.col('MeanSpeed') < 5)
print(f"number of data points={sample_data.shape[0]}")
sample_data = sample_data.unique(subset=['Epoch', 'VehID'])
# get the initial y (y0) and frame id (t0)
sample_num, act_num = 100, 0
all_initial_data = sample_data.sample(n=sample_num)
for ind in range(sample_num):
    init_data = all_initial_data[ind]
    wave_data = wave_propagation_polars(dfc, init_data, epoch_to_second=1 / 1000)
    # rename the frame to tc for joining with data
    wave_data = wave_data.rename({'Epoch': 'tc'})
    # get the average speed in nearby frames for each data row in wave_data
    platoon_speed = collect_avg_speed_polars(data, wave_data)
    speed_dev = platoon_speed - avg_speed
    # find the first index where the speed < avg_speed, threshold = 0.1 m/s
    slow_ind = np.where(speed_dev < 0)[0]
    # find the longest consecutive index starting from the first index
    slow_ind = np.split(slow_ind, np.where(np.diff(slow_ind) != 1)[0] + 1)
    longest_ind = max(slow_ind, key=len)
    # if empty, skip
    if len(longest_ind) <= 5:
        continue
    # find the first and last index of the consecutive index
    first_ind, last_ind = int(min(longest_ind)), int(max(longest_ind))
    # # plot the valid_speed_dev
    # plt.plot(valid_speed_dev, marker='o')
    valid_dis = abs(wave_data['LocalY'][last_ind] - wave_data['LocalY'][first_ind])
    if valid_dis < 20:
        continue
    act_num += 1
    # collect the speed dev for each leader-follower pair
    valid_speed_dev = speed_dev[first_ind:last_ind + 1]
    leader_dev.append(valid_speed_dev[:-1])
    follower_dev.append(valid_speed_dev[1:])
    # get the frame id and local y of the first and last index
    first_frame, last_frame = wave_data['tc'][first_ind], wave_data['tc'][last_ind]
    first_local_y, last_local_y = wave_data['LocalY'][first_ind], wave_data['LocalY'][last_ind]
    # plot the first and last index
    plt.plot([first_frame / 1000, last_frame / 1000], [first_local_y, last_local_y], marker='o', c='k', lw=1)
plt.scatter(dfc['Epoch'] / 1000, dfc['LocalY'], c=dfc['MeanSpeed'], cmap='coolwarm_r', s=2)
# invert the y-axis
plt.gca().invert_yaxis()
plt.xlabel('Time (s)')
plt.ylabel('Local Y (m)')
plt.colorbar()
plt.show()

# print the number of data points
print(f"number of data points={act_num}")
# flatten the nested list of leader_dev
leader_dev, follower_dev = itertools.chain(*leader_dev), itertools.chain(*follower_dev)
leader_dev, follower_dev = np.array(list(leader_dev)), np.array(list(follower_dev))
lr = linear_model.LinearRegression().fit(leader_dev.reshape(-1, 1), follower_dev)
slope, intercept = lr.coef_[0], lr.intercept_
# slope=0.8226758861911331, intercept=-0.6037869445793889
# lamda3 = (1 - slope) * lamda1 / slope
# slope, intercept = lamda1 / (lamda1 + lamda3), -lamda2 / (lamda1 + lamda3) / 2
print(f"slope={slope}, intercept={intercept}")
print(f"R^2={lr.score(leader_dev.reshape(-1, 1), follower_dev)}")
# plot the follower_dev vs. leader_dev
fig, ax = plt.subplots()
ax.scatter(leader_dev, follower_dev, color='b', s=15)
# plot the linear regression line and the R^2
ax.plot(leader_dev, slope * leader_dev + intercept, color='r')
# ax.set_xlim([-5, 0])
# set the box aspect equal
ax.set_box_aspect(1)
# set the text
ax.text(0.1, 0.8, f"$R^2$={lr.score(leader_dev.reshape(-1, 1), follower_dev):.1f}", transform=plt.gca().transAxes)
ax.set_xlabel('Leader speed disturbance (m/s)')
ax.set_ylabel('Follower speed disturbance (m/s)')
plt.show()
