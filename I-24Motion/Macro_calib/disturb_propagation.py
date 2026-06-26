import numpy as np
import matplotlib.pyplot as plt
import itertools
from sklearn import linear_model
from funcs_for_correlation import wave_propagation_polars, collect_avg_speed_polars
import polars as pl

plt.rcParams.update({'font.size': 30, 'font.family': 'Times New Roman'})
plt.rcParams["mathtext.fontset"] = "cm"
# load the data and the correlation length
# # "FrameID", "LaneID", "VehID", "VehClass", "Vehlen", "LocalY", "MeanSpeed"
ft_to_m = 0.3048
# unit: ft/s
data = pl.read_parquet("../data/I24_data_lane2.parquet")
# convert the ft to m
data = data.with_columns(Speed_m=pl.col('MeanSpeed') * ft_to_m, LocalY_m=pl.col('LocalY') * ft_to_m)
# filter the data that are in the isolated bottleneck (LocalY in [MM 58.8, MM59.2]*5280 ft)
data = data.filter((pl.col('LocalY') >= 58.8 * 5280) & (pl.col('LocalY') <= 59.2 * 5280))
# convert the FrameID in Unix timestamps to datetime in timezone UTC-6 CST
data = data.with_columns(((pl.from_epoch(pl.col('FrameID'))).dt.offset_by('-6h')).alias('time'))
# construct the integer frame column (data re-sampled in 25 hz)
data = data.with_columns(frame=((pl.col('FrameID') - pl.col('FrameID').min()) / 0.04).cast(pl.Int32))
# drop the duplicate rows with the same frame and vehicle ID
data = data.unique(subset=['frame', 'VehID'])
# select columns: ['frame', 'VehID', 'Speed_m', 'LocalY_m', 'time']
data = data.select(['frame', 'VehID', 'Speed_m', 'LocalY_m', 'time'])
# read the fundamental diagram file: columns=[t, q, k, v]
fd_data = pl.read_excel('../data/fd_data.xlsx')
# read the correlation file: columns=['frame', 'corr_veh', 'corr_value', 'max_veh_len', 'corr_len']
corr_data = pl.read_excel('../data/correlation_length.xlsx')
# find the data with maximum correlation length
# calculate the critical density with maximum correlation length
idc = corr_data['corr_len'].arg_max()
critical_time = corr_data[idc]['time'][0]
# group the data by time and get the idc th row
data = data.sort(by=['time'])
df = data.group_by_dynamic('time', every='1m').agg(framelist=pl.col('frame'), vehlist=pl.col('VehID'),
                                                   speedlist=pl.col('Speed_m'), localylist=pl.col('LocalY_m'))
dfc = df[idc]
# explode the list columns of dfc
dfc = dfc.explode(['framelist', 'vehlist', 'speedlist', 'localylist'])
avg_speed = dfc['speedlist'].mean()
# rename the columns to frame, VehID, Speed_m, LocalY_m
dfc = dfc.rename({'framelist': 'frame', 'vehlist': 'VehID', 'speedlist': 'Speed_m', 'localylist': 'LocalY_m'})
# filter the data in the first frame
dfc0 = dfc.filter(pl.col('frame') == pl.col('frame').min())
dfc0 = dfc0.rename({'frame': 'tc'})
# sort the data by frame and local y
dfc0 = dfc0.sort(by=['tc', 'LocalY_m'])
# get the average speed of each vehicle in the fisrt frame of the critical time window
init_speed = collect_avg_speed_polars(data, dfc0)
# calculate the speed deviation of the first 3 slow-speed vehicles
slow_ind = np.where(init_speed < avg_speed)[0]
init_disturbance = np.mean(init_speed[slow_ind[0]]) - avg_speed
print(f"initial disturbance={init_disturbance} m/s")
# initial disturbance=-1.4174327202700372 m/s
# # plot the initial disturbance
# plt.figure()
# plt.plot(dfc0['LocalY_m'], init_speed, marker='o', c='b', lw=1)
# plt.plot([dfc0['LocalY_m'].min(), dfc0['LocalY_m'].max()], [avg_speed, avg_speed], c='k', lw=1)
# plt.xlabel('Local Y (m)')
# plt.ylabel('Speed (m/s)')
# plt.show()

# get the data around the critical time
dfc = data.filter((pl.col('frame') - pl.min('frame') >= critical_time * 60 * 25 - 25 * 10) & (
        pl.col('frame') - pl.min('frame') <= (critical_time + 1) * 60 * 25 + 25 * 10))
avg_speed = dfc['Speed_m'].mean()
follower_dev, leader_dev, possible_init_disturb = [], [], []
# sample from the low-speed vehicles
sample_data = dfc.filter(pl.col('Speed_m') < 3)
sample_data = sample_data.unique(subset=['frame', 'VehID'])
# get the initial y (y0) and frame id (t0)
sample_num, act_num = 400, 0
all_initial_data = sample_data.sample(n=sample_num)
for ind in range(sample_num):
    init_data = all_initial_data[ind]
    wave_data = wave_propagation_polars(dfc, init_data, frame_to_second=1 / 25)
    # rename the frame to tc for joining with data
    wave_data = wave_data.rename({'frame': 'tc'})
    # get the average speed in nearby frames for each data row in wave_data
    platoon_speed = collect_avg_speed_polars(data, wave_data)
    speed_dev = platoon_speed - avg_speed
    # find the first index where the speed < avg_speed, threshold = 0.1 m/s
    slow_ind = np.where(speed_dev < 0)[0]
    # find the longest consecutive index starting from the first index
    slow_ind = np.split(slow_ind, np.where(np.diff(slow_ind) != 1)[0] + 1)
    longest_ind = max(slow_ind, key=len)
    # if empty, skip
    if len(longest_ind) <= 2:
        continue
    # find the first and last index of the consecutive index
    first_ind, last_ind = int(min(longest_ind)), int(max(longest_ind))
    # # plot the valid_speed_dev
    # plt.plot(valid_speed_dev, marker='o')
    valid_dis = abs(wave_data['LocalY_m'][last_ind] - wave_data['LocalY_m'][first_ind])
    if valid_dis < 20:
        continue
    act_num += 1
    # collect the speed dev for each leader-follower pair
    valid_speed_dev = speed_dev[first_ind:last_ind + 1]
    leader_dev.append(valid_speed_dev[:-1])
    follower_dev.append(valid_speed_dev[1:])
    # get the frame id and local y of the first and last index
    first_frame, last_frame = wave_data['tc'][first_ind], wave_data['tc'][last_ind]
    first_local_y, last_local_y = wave_data['LocalY_m'][first_ind], wave_data['LocalY_m'][last_ind]
    # # plot the first and last index
    # plt.plot([first_frame / 25, last_frame / 25], [first_local_y, last_local_y], marker='o', c='k', lw=1)
# plt.scatter(dfc['frame'] / 25, dfc['LocalY_m'], c=dfc['Speed_m'], cmap='coolwarm_r', s=2)
# # invert the y-axis
# plt.gca().invert_yaxis()
# plt.xlabel('Time (s)')
# plt.ylabel('Local Y (m)')
# plt.colorbar()
# plt.show()

# print the number of data points
print(f"number of data points={act_num}")
# flatten the nested list of leader_dev
leader_dev, follower_dev = itertools.chain(*leader_dev), itertools.chain(*follower_dev)
leader_dev, follower_dev = np.array(list(leader_dev)), np.array(list(follower_dev))
# if the abstract between the leader and follower is larger than 5 m/s, filter out
ind = np.where(abs(leader_dev - follower_dev) < 5)
leader_dev, follower_dev = leader_dev[ind], follower_dev[ind]
lr = linear_model.LinearRegression().fit(leader_dev.reshape(-1, 1), follower_dev)
slope, intercept = lr.coef_[0], lr.intercept_
# slope=0.9608884438215662, intercept=-0.12069488484306401
# slope=0.9608835878227541, intercept=-0.11898780828269029
# slope=0.9585873559534448, intercept=-0.1266698993137485
# slope=0.9594304442978445, intercept=-0.12286372880052454
# slope=0.9593452054153437, intercept=-0.12447977744245087
# lamda3 = (1 - slope) * lamda1 / slope
# slope, intercept = lamda1 / (lamda1 + lamda3), -lamda2 / (lamda1 + lamda3) / 2
print(f"slope={slope}, intercept={intercept}")
# plot the follower_dev vs. leader_dev
fig, ax = plt.subplots()
ax.scatter(leader_dev, follower_dev, color='b', s=20)
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
