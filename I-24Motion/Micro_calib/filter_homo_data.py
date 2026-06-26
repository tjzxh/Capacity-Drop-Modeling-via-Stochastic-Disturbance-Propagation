import numpy as np
import matplotlib.pyplot as plt
import polars as pl
from datetime import datetime

ft_to_m = 0.3048
# unit: ft/s
# "FrameID", "LaneID", "VehID", "VehClass", "Vehlen", "LocalY", "MeanSpeed"
data = pl.read_parquet("../data/I24_data_lane2.parquet")
# filter the data that are in the isolated bottleneck (LocalY in [MM 58.8, MM59.2]*5280 ft)
data = data.filter((pl.col('LocalY') >= 58.8 * 5280) & (pl.col('LocalY') <= 59.2 * 5280))
# convert the FrameID in Unix timestamps to datetime in timezone UTC-6 CST
data = data.with_columns(((pl.from_epoch(pl.col('FrameID'))).dt.offset_by('-6h')).alias('time'))
# filter the data that are in stable traffic condition (time before 6:05, 6:55-8:50, after 9:10)
data = data.filter((pl.col('time').is_between(datetime(2022, 11, 22, 7, 00),
                                              datetime(2022, 11, 22, 8, 00))))
# calculate the maximum speed
speeds = data['MeanSpeed'].to_numpy() * ft_to_m * 3.6
point_y = 85
point_x = np.percentile(speeds, point_y)
print("Maximum speed: {} km/h".format(point_x))  # 102.56190352281752 km/h

# calculate the proportion of auto (VehClass=0) in the data
print("Proportion of autos: {}%".format(
    100 * np.sum(data['VehClass'].to_numpy() == 0) / len(data['VehClass'].to_numpy())))  # 42.59884263637462%

# filter the data using polars group_by
# [frame, subject speed, leader speed, speed difference, average speed in lane, spacing, subject class, relative class]
# construct the integer frame column (data re-sampled in 25 hz)
data = data.with_columns(frame=((pl.col('FrameID') - pl.col('FrameID').min()) / 0.04).cast(pl.Int32))
print("There are {} data in total.".format(data.shape[0]))
# # cast the VehID column to string
# data = data.with_columns(VehID=pl.col('VehID').cast(pl.Utf8))
# drop the duplicate rows with the same frame and vehicle ID
data = data.unique(subset=['frame', 'VehID'])
print("There are {} data after dropping the duplicate rows.".format(data.shape[0]))
# filter the vehicles that are in the data for more than 60 seconds
data_for_CF = data.group_by('VehID').agg(CF_time_list=pl.col('FrameID'))
data_for_CF = data_for_CF.with_columns(CF_time=pl.col('CF_time_list').list.max() - pl.col('CF_time_list').list.min())
threshold_time = 10  # time for stable car-following
valid_vID = data_for_CF.filter(pl.col('CF_time') > threshold_time)['VehID']

# sort the data by time and LocalY
data = data.sort(by=['LocalY'])
df = data.group_by('frame', maintain_order=True).agg(pl.col('MeanSpeed'), pl.col('LocalY'), pl.col('Vehlen'),
                                                     pl.col('VehClass'), pl.col('VehID'))
df = df.with_columns(
    meanSpeed=pl.col('MeanSpeed').list.mean(), subjectSpeed=pl.col('MeanSpeed').list.slice(1, None),
    leaderSpeed=pl.col('MeanSpeed').list.slice(0, pl.col('MeanSpeed').list.len() - 1),
    speedDiff=pl.col('MeanSpeed').list.diff(null_behavior='drop'),
    spacing=pl.col('LocalY').list.diff(null_behavior='drop'),
    subjectClass=pl.col('VehClass').list.slice(1, None),
    relativeClass=pl.col('VehClass').list.diff(null_behavior='drop'), Vehlength=pl.col('Vehlen').list.slice(1, None),
    vID=pl.col('VehID').list.slice(1, None))
# select the columns: FrameID, subjectSpeed, leaderSpeed, speedDiff, spacing, subjectClass, relativeClass
dff = df.select(
    ['frame', 'subjectSpeed', 'leaderSpeed', 'speedDiff', 'spacing', 'subjectClass', 'relativeClass', 'Vehlength',
     'meanSpeed', 'vID'])
# drop the empty column of spacing
dff = dff.drop_nulls()
# explode the data
df_explode = dff.explode(
    ['subjectSpeed', 'leaderSpeed', 'speedDiff', 'spacing', 'subjectClass', 'relativeClass', 'Vehlength', 'vID'])
# net spacing = spacing - length of subject vehicle
df_explode = df_explode.with_columns(net_spacing=pl.col('spacing') - pl.col('Vehlength'))
# filter the data for valid vehicles
df_explode = df_explode.filter(pl.col('vID').is_in(valid_vID))
print("There are {} vehicles before filtering and {} vehicles that appear for more than {} s.".format(len(data_for_CF),
                                                                                                      len(valid_vID),
                                                                                                      threshold_time))
print("There are {} data after filtering the vehicles that are in the data for more than {} seconds.".format(
    df_explode.shape[0], threshold_time))

# filter the data for homogeneous: append only for sedan (classID=0 and relative class=0)
homo_data = df_explode.filter((pl.col('subjectClass') == 0) & (pl.col('relativeClass') == 0))
homo_data = homo_data.filter(pl.col('net_spacing') > 0)
# # debug the negative spacing
# bad = homo_data.filter(pl.col('net_spacing') < 0)
# print('There are {} negative spacing.'.format(bad.shape[0]))  # There are 0 negative spacing.
# save the data
homo_data.write_parquet('../data/I24_lane2_homo_data.parquet')
df_explode.write_parquet('../data/I24_lane2_all_data.parquet')
print("Done! There are total {} data points and {} homo data points.".format(df_explode.shape[0], homo_data.shape[0]))
# There are total 4667577 data points and 912917 homo data points.
