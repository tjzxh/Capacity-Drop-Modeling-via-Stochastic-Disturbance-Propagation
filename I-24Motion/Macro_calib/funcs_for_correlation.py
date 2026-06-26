import numpy as np
from scipy import stats
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
from itertools import chain
import polars as pl


# define the wave propagation function: given the initial disturbance, find the following disturbance
def wave_propagation(data, initial_ind, frame_to_second, w=20 / 3.6):
    # get the initial y (y0) and frame id (t0)
    initial_data = data.loc[initial_ind]
    initial_y, initial_frame = initial_data['LocalY'], initial_data['FrameID']
    # filter the possible data with t!=t0 to avoid divide by zero
    data = data[data['FrameID'] != initial_frame]
    # find the data that (y0-y)/(t-t0) match w best except the initial disturbance
    possible_data = data[
        np.abs((initial_y - data['LocalY'].values) / ((initial_frame - data['FrameID'].values) * frame_to_second) - (
            - w)) < 0.1]
    # check each veh_id only appears once, if not, keep the first one
    possible_data = possible_data.drop_duplicates(subset=['VehID'], keep='first')
    # add the initial disturbance to the possible data
    possible_data = pd.concat([initial_data, possible_data])
    # sort the possible data by the frame id (descending)
    possible_data = possible_data.sort_values(by=['FrameID'], ascending=False)
    return possible_data


# for a given data of a vehicle, collect the speed of nearby frames and calculate the average speed
def collect_avg_speed(data, given_data, frame_window=100):
    # get the frame id of the given index
    given_frame, veh_id = given_data['FrameID'], given_data['VehID']
    # get the data of the nearby frames for the same vehicle
    nearby_frame_data = data[(data['VehID'] == veh_id) & (data['FrameID'] >= given_frame - frame_window) & (
            data['FrameID'] <= given_frame + frame_window)]
    # calculate the average speed
    mean_speed = nearby_frame_data['MeanSpeed'].mean()
    return mean_speed


def wave_propagation_polars(data, initial_data, frame_to_second, w=20 / 3.6):
    initial_y, initial_frame = initial_data['LocalY_m'][0], initial_data['frame'][0]
    # filter the possible data with t!=t0 to avoid divide by zero
    df = data.filter(pl.col('frame') != initial_frame)
    # find the data that (y0-y)/(t-t0) match w best except the initial disturbance (if I-24 MOTION, w; others, -w)
    possible_data = df.filter(
        ((initial_y - pl.col('LocalY_m')) / ((initial_frame - pl.col('frame')) * frame_to_second) - (w)).abs() < 0.1)
    # add the initial disturbance to the possible data
    all_data = pl.concat([initial_data, possible_data])
    # check each veh_id only appears once, if not, keep the first one
    all_data = all_data.unique(['frame', 'VehID'])
    # sort the possible data by the frame id
    all_data = all_data.sort(by=['frame', 'LocalY_m'])
    return all_data


def collect_avg_speed_polars(data, dfc, frame_interval=10 * 25):
    # expected dfc columns: tc, VehID (unique), Speed_m, LocalY_m
    dfc = dfc.select('tc', 'VehID')
    # filter the data that VehID is in dfc
    data = data.filter(pl.col('VehID').is_in(dfc['VehID']))
    # join the data with dfc on VehID
    data = data.join(dfc, on='VehID', how='inner')
    # compute the average speed when the frame is in the interval of [tc-frame_interval, tc+frame_interval]
    nearby_data = data.group_by('VehID').agg(aggV=pl.when((pl.col('frame') >= pl.col('tc') - frame_interval) & (
            pl.col('frame') <= pl.col('tc') + frame_interval)).then(pl.mean('Speed_m')),
                                             aggY=pl.when((pl.col('frame') >= pl.col('tc') - frame_interval) & (
                                                     pl.col('frame') <= pl.col('tc') + frame_interval)).then(
                                                 pl.mean('LocalY_m')))
    nearby_data = nearby_data.with_columns(avgV=pl.col('aggV').list.mean(), avgY=pl.col('aggY').list.mean())
    # join the nearby_data with dfc on VehID
    nearby_data = nearby_data.join(dfc, on='VehID', how='inner')
    # sort the nearby_data by avgY
    nearby_data = nearby_data.sort(by=['tc', 'avgY'])
    avg_speed = nearby_data['avgV'].to_numpy()
    return avg_speed


# collect all the speed data for each frame
def collect_speed(df, time_para, lane):
    min_time, max_time, time_window, time_slip, num, fts = time_para
    df = df[(df.FrameID >= min_time) & (df.FrameID <= max_time) & (df.LaneID == lane)]
    # iterate over the FrameID with 10 intervals
    all_frame_speed, all_frame, all_frame_pos = [], df.FrameID.unique(), []
    np.sort(all_frame)
    tg = 25 if fts == 1 else int(1 / fts)
    all_frame = all_frame[::tg]
    for frame in all_frame:
        # get all data for this frame and sort by LocalY
        df_frame = df[df.FrameID == frame].sort_values(by=['LocalY'])
        # get the structured mean speed
        lane_speed_data = df_frame['MeanSpeed'].values
        all_frame_speed.append(lane_speed_data)
        # get the localY
        lane_pos_data = df_frame['LocalY'].values
        all_frame_pos.append(lane_pos_data)
    return all_frame_speed, all_frame_pos


# correlation for one lane
def correlate(df, n, all_frame_speed, all_frame_pos, time_para, lane, space_para):
    min_time, max_time, time_window, time_slip, num, fts = time_para
    _, _, min_y_cl, max_y_cl, _ = space_para
    # directly give all_corr=1 and all_dis=0 for n=0
    if not n:
        return [1] * num, [0] * num
    df = df[(df.LocalY >= min_y_cl) & (df.LocalY <= max_y_cl) & (df.FrameID >= min_time) & (df.FrameID <= max_time) & (
            df.LaneID == lane)]
    # iterate over the FrameID
    all_frame = df.FrameID.unique()
    np.sort(all_frame)
    tg = 25 if fts == 1 else int(1 / fts)
    all_frame = all_frame[::tg]
    # calculate the correlation coefficient by aggregating every time window and move on with time slip
    all_corr, all_dis = [], []
    for i in range(num):
        # get all data for this time window
        ind = (all_frame >= min_time + i * time_slip) & (all_frame <= min_time + i * time_slip + time_window)
        ind = ind.nonzero()[0]
        # calculate the average spacing for n vehicles
        posx = [all_frame_pos[k][:-n] if len(all_frame_pos[k]) > n else [] for k in ind]
        posy = [all_frame_pos[k][n:] if len(all_frame_pos[k]) > n else [] for k in ind]
        posx, posy = list(chain.from_iterable(posx)), list(chain.from_iterable(posy))
        distance_for_n = np.abs(np.subtract(posy, posx))
        all_dis.append(np.mean(distance_for_n))
        # calculate the speed correlation
        datax = [all_frame_speed[j][:-n] if len(all_frame_speed[j]) > n else [] for j in ind]
        datay = [all_frame_speed[j][n:] if len(all_frame_speed[j]) > n else [] for j in ind]
        # flatten the datax and datay
        datax, datay = list(chain.from_iterable(datax)), list(chain.from_iterable(datay))
        if len(datax) < 2:
            corr = 0
        else:
            corr = stats.pearsonr(datax, datay)[0]
        all_corr.append(corr)
    return all_corr, all_dis


# check the correlation length, return correlation length (index) and flag (0/1/2 for no/left/right sign change)
def check_corr_min(valid_corr):
    max_veh_len = len(valid_corr) - 1
    corr_min_value, corr_min_index = np.min(np.abs(valid_corr)), np.argmin(np.abs(valid_corr))
    if max_veh_len <= 3:
        return corr_min_index, 0
    # detect whether the valid_corr is a list, if is, convert to array
    if isinstance(valid_corr, list):
        valid_corr = np.array(valid_corr)
    # detect where the sign changes
    possible_ind_list = np.where(valid_corr[:-1] * valid_corr[1:] < 0)[0]
    sorted_ind_list = np.sort(possible_ind_list)
    # if there is no sign change and positive for all, return the maximum index
    if not sorted_ind_list.size:
        if corr_min_value > 0.1:
            return max_veh_len, 0
        else:
            return corr_min_index, 0
    elif abs(valid_corr[sorted_ind_list[0]]) < abs(valid_corr[sorted_ind_list[0] + 1]):
        return sorted_ind_list[0], 1
    else:
        return sorted_ind_list[0] + 1, 2


# compute the q,k,v from the data using the Edie's method
def cal_fd(df, time_para, lane, space_para):
    min_time, max_time, time_window, time_slip, num, frame_to_second = time_para
    min_y_fd, max_y_fd, _, _, total_veh = space_para
    df = df[(df.FrameID >= min_time) & (df.FrameID <= max_time) & (df.LocalY >= min_y_fd) & (
            df.LocalY <= max_y_fd) & (df.LaneID == lane)]
    fd_data = []
    # calculate the area of the rectangle
    area = time_window * frame_to_second * (max_y_fd - min_y_fd)
    for i in range(num):
        all_dis, all_time = [], []
        # get all data for this time window
        df_time = df[(df.FrameID >= min_time + i * time_slip) & (df.FrameID <= min_time + i * time_slip + time_window)]
        # iterate over all vehicles
        for veh_id in list(set(df_time['VehID'])):
            # get all data for this vehicle
            veh_data = df_time.loc[df_time['VehID'] == veh_id]
            # get all local y and frame id
            local_y = veh_data['LocalY']
            frame_id = veh_data['FrameID']
            # calculate the distance and time travelled by this vehicle
            dis = max(local_y) - min(local_y)
            time = max(frame_id) - min(frame_id)
            # save the distance and the time
            all_dis.append(dis)
            all_time.append(time)
        # derive the flow and density in this time window
        q = sum(all_dis) * 3600 / area
        k = sum(all_time) * frame_to_second * 1000 / area
        v = q / k if k else 0
        fd_data.append([q, k, v])
    # save the data into Excel with headers: q (veh/h), k (veh/km), v (km/h)
    fd_data = np.array(fd_data)
    fd_data = pd.DataFrame(fd_data, columns=['q', 'k', 'v'])
    fd_data.to_excel('fd_lane' + str(lane) + '.xlsx', index=False)


# parse the data and return the data, time_para, space_para, v_thre
def parse_data(dataset='I80'):
    if dataset == 'I80':
        ##### I-80 dataset parameters #####
        min_time, max_time, time_window, time_slip, frame_to_second = 10, 9000 - 10, 600, 100, 0.1  # I80
        min_y_fd, max_y_fd = 50, 250  # space range for fundamental diagram
        total_veh, v_thre = 30, 8  # I80
        # reading the reconstructed I-80 data from the file
        data = pd.read_csv("D:/Reconstructed_I80/reconstructed_NGSIM.txt", sep='\s+', header=None,
                           names=["VehID", "FrameID", "LaneID", "LocalY", "MeanSpeed", "MeanAcc",
                                  "Vehlen", "VehClass", "FollowerID", "LeaderID"])
    elif dataset == 'US101':
        ##### US-101 dataset parameters #####
        min_time, max_time, time_window, time_slip, frame_to_second = 10, 8500 - 100, 100, 100, 0.1  # US101
        min_y_fd, max_y_fd = 200, 400  # US101
        total_veh, v_thre = 41, 10  # US101
        # read the US-101 data from the csv file with headers, change the ft to m
        data = pd.read_csv("D:/US101/trajectories-0750am-0805am.csv", sep=",", header=0,
                           names=["VehID", "FrameID", "TotalFrame", "GlobalTime", "LocalX", "LocalY", "GX", "GY",
                                  "Vehlen",
                                  "Vw", "VehClass", "MeanSpeed", "MeanAcc", "LaneID", "LeaderID", "FollowerID",
                                  "space_hw",
                                  "time_hw"])
        data['MeanSpeed'], data['LocalY'] = data['MeanSpeed'] * 0.3048, data['LocalY'] * 0.3048
    elif dataset == 'HighD':
        ##### HighD parameters #####
        min_time, max_time, time_window, time_slip, frame_to_second = 10, 9000, 100, 100, 0.1  # HighD
        min_y, max_y = 60, 160  # HighD
        total_veh = 31  # HighD
        # read text file into pandas DataFrame and create header with names, units:ft
        data = pd.read_csv("./36_tracks.csv", sep=',', header=0,
                           names=["FrameID", "VehID", "LocalY", "LateralPosition", "Vehlen", "Vehwidth", "MeanSpeed",
                                  "LocalYVelocity", "Acc", "LocalYAcc", "FrontSightDistance",
                                  "BackSightDistance", "DHW",
                                  "THW", "TTC", "LeaderSpeed", "PrecedingID", "FollowingID", "LeftPrecedingID",
                                  "LeftAlongsideID",
                                  "LeftFollowingID", "RightPrecedingID", "RightAlongsideID", "RightFollowingID",
                                  "LaneID"])
    elif dataset == 'SQM':
        ##### SQM parameters #####
        min_time, max_time, time_window, time_slip, frame_to_second = 100, 8000 - 100, 24 * 5, 24 * 5, 1 / 24
        min_y, max_y = 10, 370
        total_veh = 30
        # read xlsx file into pandas DataFrame and create header with names, VehicleID	Time(Frame)	Time(s)	LaneID	LongtitudePosition(pixel)	LongtitudePosition(meter)	Gap(pixel)	Gap(meter)	Speed(m/s)	Speed(km/h)	Acceleration(m/s^2)	LatitudePostion(Pixel)	LatitudePosition(meter)	VehicleLength(pixel)	VehicleWidth(pixel)
        data = pd.read_excel("D:/SEU/SQM1.xlsx", sheet_name="Sheet1", header=0,
                             names=["VehID", "FrameID", "Time", "LaneID", "LongtitudePosition(pixel)",
                                    "LocalY", "Gap(pixel)", "Gap(meter)", "MeanSpeed", "Speed(km/h)",
                                    "Acc", "LatitudePostion(Pixel)", "LatitudePosition(meter)",
                                    "VehicleLength(pixel)", "VehicleWidth(pixel)"])
    elif dataset == 'I405':
        ##### I-405 dataset parameters #####
        min_time, max_time, time_window, time_slip, frame_to_second = 0, 3600, 60, 60, 1
        min_y_fd, max_y_fd = 10, 450  # m
        total_veh = 30
        # read the I405 data from the DAT file
        data = pd.read_csv("D:/I405/SANTAMON.csv", header=None, sep=',',
                           names=["FrameID", "VehID", "VehClass", "Vehlen", "MeanSpeed", "LocalY", "LocalX", "Color",
                                  "LaneID"])
        data['MeanSpeed'], data['LocalY'] = 1.60934 * data['MeanSpeed'] / 3.6, data['LocalY'] * 0.3048  # m/s, m
    elif dataset == 'I24':
        ##### I-24 dataset parameters #####
        # convert date to unix time
        # min_time, max_time = 1669118400, 1669132800
        time_window, time_slip, frame_to_second = 60, 60, 1
        min_y_fd, max_y_fd = 59.0 * 5280 * 0.3048, 59.0 * 5280 * 0.3048 + 200  # ft to m
        min_y_cl, max_y_cl = 58.7 * 5280 * 0.3048, 59.5 * 5280 * 0.3048  # ft to m
        total_veh = 40
        # read the I-24 lane 2 data from the json file with headers:timestamp, lane_id, veh_id, veh_class, length, x_position, x_speed
        # data = pd.read_json("./data/I24_data_lane2.json")
        # data['timestamp'] = data['timestamp'].map(lambda x: x.replace(tzinfo=datetime.timezone.utc).timestamp())
        # data.astype(float)
        # data.to_csv("./I24_data_lane2.csv", index=False)
        # change the headers to the same as other datasets
        data = pd.read_csv("./I24_data_lane2.csv", header=0,
                           names=["FrameID", "LaneID", "VehID", "VehClass", "Vehlen", "LocalY", "MeanSpeed"])
        # change the unit to m
        data['MeanSpeed'], data['LocalY'] = data['MeanSpeed'] * 0.3048, data['LocalY'] * 0.3048
        min_time, max_time = min(data['FrameID']), max(data['FrameID'])

    # maximum number of time windows
    num = int((max_time - time_window - min_time) // time_slip)
    time_para = [min_time, max_time, time_window, time_slip, num, frame_to_second]
    space_para = [min_y_fd, max_y_fd, min_y_cl, max_y_cl, total_veh]
    return data, time_para, space_para


# calculate the correlation coefficient C(n,t)
def main_cl(df, lane, time_para, space_para):
    min_time, max_time, time_window, time_slip, num, _ = time_para
    min_y_fd, max_y_fd, min_y_cl, max_y_cl, total_veh = space_para
    corr_array, dis_array = np.zeros((num, total_veh)), np.zeros((num, total_veh))
    all_frame_speed, all_frame_pos = collect_speed(df, time_para, lane)
    for j in range(total_veh):
        corr, dis = correlate(df, j, all_frame_speed, all_frame_pos, time_para, lane, space_para)
        print(f"Correlation for n={j} is calculated")
        if not corr:
            continue
        corr_array[:, j] = np.array(corr)
        dis_array[:, j] = np.array(dis)
    # save to excel and convert to array back
    corr_array = pd.DataFrame(corr_array, columns=['n=' + str(i) for i in range(total_veh)])
    dis_array = pd.DataFrame(dis_array, columns=['n=' + str(i) for i in range(total_veh)])
    with pd.ExcelWriter('coef_lane' + str(lane) + '.xlsx') as writer:
        corr_array.to_excel(writer, sheet_name='C(t,n)', index=False)
        dis_array.to_excel(writer, sheet_name='s(t,n)', index=False)
    corr_array, dis_array = corr_array.to_numpy(), dis_array.to_numpy()

    # calculate the correlation length, i.e., find the first n that for each t, C(r,t) nearest to zero
    all_corr_len = []
    for i in range(num):
        sub_corr, sub_dis = corr_array[i, :], dis_array[i, :]
        valid_corr = np.trim_zeros(sub_corr, 'b')
        # check the correlation length
        if not valid_corr.shape[0]:
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
            f"Time: {min_time + i * time_slip}, corr vehicles: {checked_corr_min_index}, corr value: {checked_corr_min_value}, corr length : {correlation_length}")
        # check correlation length is nan
        if np.isnan(correlation_length):
            # show the correlation coefficient C(n) for each t
            plt.plot(np.arange(len(valid_corr)), valid_corr, lw=1, marker='.',
                     label='t=' + str(min_time + i * time_slip),
                     color='b')
            plt.plot(range(total_veh), [0] * total_veh, ls='--', lw=1, color='r')
            plt.xlim((0, total_veh - 1))
            plt.legend()
            plt.show(block=True)
            print('Correlation length is nan')
        all_corr_len.append(
            [min_time + i * time_slip, checked_corr_min_index, checked_corr_min_value, max_veh_len, correlation_length])
    # save the all_corr_len to excel
    all_corr_len = np.array(all_corr_len)
    all_corr_len = pd.DataFrame(all_corr_len,
                                columns=['frame', 'corr_veh', 'corr_value', 'max_veh_len', 'corr_len'])
    all_corr_len.to_excel('correlation_lane' + str(lane) + '.xlsx', index=False)
