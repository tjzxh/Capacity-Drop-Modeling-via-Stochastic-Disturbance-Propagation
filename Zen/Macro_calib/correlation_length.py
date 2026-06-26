import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from funcs_for_correlation import check_corr_min
import polars as pl
import scipy.stats as stats

# unit: m, m/s
# # "FrameID", "LaneID", "VehID", "VehClass", "Vehlen", "LocalY", "MeanSpeed"
data = pl.read_parquet("../data/Zen_data.parquet")
# sort the data by FrameID
data = data.sort(by=['FrameID'])
data = data.set_sorted(column=['time'])
# filter the data that are in the isolated bottleneck (LocalY in [KP 3.6, KP 4.2]*1000 m)
data = data.filter((pl.col('LocalY') >= 3650) & (pl.col('LocalY') <= 4200))
data = data.unique(subset=['FrameID', 'VehID'])
data = data.sort(by=['LocalY'])


# calculate the pearson correlation coefficient between the vehicles' speeds that n vehicles in between every 1 minute
def correlate_for_polars(data, n):
    # aggregate the data by frame
    df = data.group_by('FrameID', maintain_order=True).agg(fpSpeed=pl.col('MeanSpeed'), fpLocalY=pl.col('LocalY'),
                                                           t=pl.col('time').unique())
    # calculate the distance between the vehicles that n vehicles in between for each frame
    df = df.with_columns(all_corr_dis=pl.when(pl.col('fpLocalY').list.len() > n).then(pl.col('fpLocalY').list.diff(n)))
    # calculate the subject speed and leader speed that n vehicles in between for each frame
    df = df.with_columns(
        all_vs=pl.when(pl.col('fpSpeed').list.len() > n).then(pl.col('fpSpeed').list.slice(n, None)),
        all_vl=pl.when(pl.col('fpSpeed').list.len() > n).then(
            pl.col('fpSpeed').list.slice(0, pl.col('fpSpeed').list.len() - n)))
    # drop the null rows
    df = df.drop_nulls(subset=['all_corr_dis', 'all_vs', 'all_vl'])
    # sort the data by time
    df = df.with_columns(time=pl.col('t').list.first())
    df = df.sort(by=['time', 'FrameID'])
    # ATTENTION: the time_interval should be consistent with the time_interval in fd_data.py
    time_interval = 60  # s
    # concat the framePlatoon column to 1 column every 30 seconds
    agg_df = df.group_by_dynamic('time', every=str(time_interval) + 's').agg(aggVs=pl.concat_list('all_vs').flatten(),
                                                                             aggVl=pl.concat_list('all_vl').flatten(),
                                                                             aggY=pl.concat_list(
                                                                                 'all_corr_dis').flatten())
    agg_df = agg_df.sort(by=['time'])
    # compute the average distance between the vehicles that n vehicles in between for each time interval
    agg_df = agg_df.with_columns(corr_dis=pl.col('aggY').list.mean())
    # calculate the correlation coefficient between leader speed and subject speed via stats.pearsonr
    leader_speed, subject_speed = agg_df['aggVl'].to_list(), agg_df['aggVs'].to_list()
    # if n is 0, set the corr_coef to 1
    corr_coef = [stats.pearsonr(leader_speed[i], subject_speed[i])[0] if len(leader_speed[i]) >= 2 else 0
                 for i in range(len(leader_speed))] if n else [1] * len(leader_speed)
    # cast the corr_coef to float64
    corr_coef = np.array(corr_coef).astype(np.float64)
    # add the corr_coef column to agg_df
    agg_df = agg_df.with_columns(corr_coef=pl.Series(corr_coef))
    return agg_df


# calculate the correlation length, i.e., find the first n that for each t, C(r,t) nearest to zero
def correlation_length_for_polars(data, max_n):
    # for each n, calculate the correlation coefficient and average distance,
    # then concat these dataframes horizontally with different names
    corr_df0 = correlate_for_polars(data, 0)
    corr_df = corr_df0
    corr_df0 = corr_df0.select(['time', 'corr_coef', 'corr_dis']).rename(
        {'corr_coef': 'corr_coef' + str(0), 'corr_dis': 'corr_dis' + str(0)})
    for n in range(1, max_n):
        corr_df = correlate_for_polars(data, n)
        corr_df = corr_df.select(['corr_coef', 'corr_dis']).rename(
            {'corr_coef': 'corr_coef' + str(n), 'corr_dis': 'corr_dis' + str(n)})
        corr_df = pl.concat([corr_df0, corr_df], how='horizontal')
        corr_df0 = corr_df
    # concat the corr_coef and corr_dis for each n to a single list vertically
    dff = corr_df.with_columns(all_corr_coef=pl.concat_list([f'corr_coef{i}' for i in range(max_n)]),
                               all_corr_dis=pl.concat_list([f'corr_dis{i}' for i in range(max_n)]))
    # drop None in the all_corr_dis and all_corr_coef list columns
    dff = dff.drop_nulls(subset=['all_corr_dis', 'all_corr_coef'])
    dff = dff.with_columns(valid_corr_coef=pl.col('all_corr_coef').list.drop_nulls(),
                           valid_corr_dis=pl.col('all_corr_dis').list.drop_nulls())
    # convert the corr_coef_n  to numpy array
    corr_array = dff.get_column('valid_corr_coef').to_numpy()
    dis_array = dff.get_column('valid_corr_dis').to_numpy()
    # save two arrays
    dff.select(['time', 'valid_corr_coef']).write_excel('../data/all_corr_coef.xlsx')
    dff.select(['time', 'valid_corr_dis']).write_excel('../data/all_corr_dis.xlsx')

    # num is the number of time intervals
    num = len(corr_array)
    # calculate the correlation length, i.e., find the first n that for each t, C(r,t) nearest to zero
    all_corr_len = []
    for i in range(num):
        sub_corr, sub_dis = corr_array[i], dis_array[i]
        valid_corr = sub_corr
        # check the correlation length
        if not len(valid_corr):
            max_veh_len, checked_corr_min_index, checked_corr_min_value, correlation_length = 0, 0, 0, 0
        else:
            max_veh_len = len(valid_corr) - 1
            checked_corr_min_index, sign_flag = check_corr_min(valid_corr)
            checked_corr_min_value = valid_corr[checked_corr_min_index]
            correlation_length = sub_dis[checked_corr_min_index]
            # average the distance when checked_corr_min_value is far from 0
            if sign_flag and abs(checked_corr_min_value) > 0.1:
                left_ind = checked_corr_min_index if sign_flag == 1 else checked_corr_min_index - 1
                right_ind = left_ind + 1
                left_corr, right_corr = abs(sub_corr[left_ind]), abs(sub_corr[right_ind])
                left_dis, right_dis = sub_dis[left_ind], sub_dis[right_ind]
                # calculate the dis across the 0
                correlation_length = left_dis + (right_dis - left_dis) / (1 + right_corr / left_corr)

        print(
            f"Time: {i}th time interval, corr vehicles: {checked_corr_min_index}, corr value: {checked_corr_min_value}, "
            f"corr length : {correlation_length}")
        # # check correlation length is nan
        # if True:
        #     # show the correlation coefficient C(n) for each t
        #     plt.plot(np.arange(len(valid_corr)), valid_corr, lw=1, marker='.',
        #              label='t=' + dff.get_column('time')[i].strftime('%H:%M:%S'),
        #              color='b')
        #     plt.plot(range(max_n), [0] * max_n, ls='--', lw=1, color='r')
        #     plt.xlim((0, max_n - 1))
        #     plt.legend()
        #     plt.show(block=True)
        #     # print('Correlation length is nan')
        all_corr_len.append(
            [dff.get_column('time')[i].strftime("%H:%M:%S"), checked_corr_min_index,
             checked_corr_min_value, max_veh_len, correlation_length])
    # save the all_corr_len to excel
    all_corr_len = np.array(all_corr_len)
    all_corr_len = pd.DataFrame(all_corr_len,
                                columns=['time', 'corr_veh', 'corr_value', 'max_veh_len', 'corr_len'])
    all_corr_len.to_excel('../data/correlation_length.xlsx', index=False)


# calculate the correlation length
# correlation_length_for_polars(data, 40)
# read the correlation file: columns=['FrameID', 'corr_veh', 'corr_value', 'max_veh_len', 'corr_len']
corr_data = pl.read_excel('../data/correlation_length.xlsx')
# read the FD data from the fd_data: t, q (veh/h), k (veh/km), v (km/h)
fd_data = pl.read_excel('../data/fd_data.xlsx')
# rename the column time to t
corr_data = corr_data.rename({'time': 't'})
# join the corr_data and fd_data
df = fd_data.join(corr_data, on='t', how='inner')
# df = pl.concat([corr_data, fd_data], how='horizontal')
# # filter the abnormal data (k>40 and corr_len>450) & (k>55)
# df = df.filter(pl.col('k') < 55)
# df = df.filter((~((pl.col('k') > 40) & (pl.col('corr_len') > 450))))
# calculate the critical density with maximum correlation length
idc = df['corr_len'].arg_max()
critical_row = df[idc]
max_corr_len = critical_row['corr_len'][0]
max_corr_veh = critical_row['corr_veh'][0]
kc = critical_row['k'][0]
print(f'The critical density is {kc} veh/km, the maximum correlation length is {max_corr_len} m '
      f'and the maximum number of correlated vehicles is {max_corr_veh}')
# The critical density is 28.2 veh/km, the maximum correlation length is 428.5512539184953 m
# and the maximum number of correlated vehicles is 12

# analyze the relationship between the correlation length and the q,k,v
plt.rcParams.update({'font.size': 30, 'font.family': 'Times New Roman'})
plt.rcParams["mathtext.fontset"] = "cm"
# subplots: 1. corr_dis vs. k 2. q vs. k 3. corr_veh vs. k 4. q vs. k
_, ax1 = plt.subplots(2, 2, sharex=True, subplot_kw=dict(box_aspect=0.7))
ax1[0, 0].scatter(df['k'], df['corr_len'], color='b')
ax1[0, 0].set_ylabel('Correlation length (m)')
ax1[1, 0].scatter(df['k'], df['q'], color='b')
ax1[1, 0].set_xlabel('Density (veh/km)')
ax1[1, 0].set_ylabel('Flow (veh/h)')
ax1[0, 1].scatter(df['k'], df['corr_veh'], color='b')
ax1[0, 1].set_ylabel('Correlation length (veh)')
ax1[1, 1].scatter(df['k'], df['q'], color='b')
ax1[1, 1].set_xlabel('Density (veh/km)')
ax1[1, 1].set_ylabel('Flow (veh/h)')
# plot the correlation length vs. k
_, ax2 = plt.subplots(1, 1)
ax2.scatter(df['k'], df['corr_len'], color='b')
# plot the theoretical correlation length lnτ⁄k[lnλ_1-ln(λ_1+λ_3)]
continuous_k = np.linspace(0, 60, 100) / 1000
tau, a1, b1, l3 = 0.0326, 0.1716, 0.7681, 0.051
l1 = a1 * np.log(continuous_k) + b1
theory_corr_len = np.log(tau) / continuous_k / (np.log(l1) - np.log(l1 + l3))
ax2.plot(continuous_k * 1000, theory_corr_len, color='r')
ax2.set_xlabel('Density (veh/km)')
ax2.set_ylabel('Correlation length (m)')
ax2.set_box_aspect(1)
plt.show()
