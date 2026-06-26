# read the trajectory dataset from D:\11-22-2022\11-22-2022\637c399add50d54aa5af0cf4__post2
# a json file contains the trajectory data of each vehicle
import pandas as pd
import polars as pl

# # use polars to read the json file to speed up the reading process
# # schema: _id, timestamp, first_timestamp, last_timestamp, x_position, y_position, length, width, height, merged_ids,
# # fragment_ids, starting_x, ending_x, coarse_vehicle_class, direction, compute_node_id, local_fragment_id, fine_vehicle_class,
# # road_segment_ids, configuration_id, flags, x_score, y_score, feasibility
# data = pl.from_pandas(
#     pd.read_json("D:/11-22-2022/11-22-2022/637c399add50d54aa5af0cf4__post2/637c399add50d54aa5af0cf4__post2.json",
#                  convert_dates=False))
#
# # Data attributes for a single vehicle trajectory:
# # veh_id,Vehicle class int – 0: sedan, 1: midsize, 2: pickup, 3: van, 4: semi, 5: truck, 6: motorcycle
# # First timestamp, Last timestamp , Timestamp [float] s Array of times at which vehicle positions are recorded
# # x position [float] ft, y position [float] ft
# # Starting x ft, Ending x ft, Length ft, Width, Height, Direction: −1 if westbound, 1 if eastbound
#
# ############ First round: filter the data in westbound(direction -1) and x_position in the range of [:,60.5*5280]
# data = data.filter((pl.col('direction') == -1) & (pl.col('ending_x') <= 60 * 5280))
# data.write_parquet("./data/I24_data.parquet")
# print("The number of vehicles in the dataset is: ", data.shape[0])
############ Second round: filter the data for varying y_position for each lane ( 12 - 24, lane 1 (HOV lane);24 - 36, lane 2; 36 - 48, lane 3; 48 - 60, lane 4)
data = pl.read_parquet("./data/I24_data.parquet")
# convert the 12-byte BSON _id to int
data = data.unnest('_id')
# select necessary columns: $oid, timestamp, x_position, y_position, length, coarse_vehicle_class
data = data.select(['$oid', 'timestamp', 'x_position', 'y_position', 'length', 'coarse_vehicle_class'])
# compute the speed at each timestamp for each vehicle and add the speed column to the data frame
data = data.with_columns((pl.col('x_position').list.diff()).alias("x_position_diff"))
data = data.with_columns((pl.col('timestamp').list.diff()).alias("timestamp_diff"))
# explode all the list columns: timestamp, x_position, y_position, x_position_diff, timestamp_diff
df = data.explode(['timestamp', 'x_position', 'y_position', 'x_position_diff', 'timestamp_diff'])
# filter the null values in the x_speed column
df = df.filter(pl.col('timestamp_diff').is_not_null())
# compute the speed at each timestamp for each vehicle
df = df.with_columns(((pl.col('x_position_diff') / pl.col('timestamp_diff')).abs()).alias("x_speed"))
df = df.filter(pl.col('x_speed').is_not_null())
# filter the data for each lane and add the lane_id column
df_lane1 = df.filter((pl.col('y_position') >= 12) & (pl.col('y_position') <= 24))
df_lane1 = df_lane1.with_columns(pl.Series("lane_id", pl.Series([1] * df_lane1.shape[0])))
df_lane2 = df.filter((pl.col('y_position') >= 24) & (pl.col('y_position') <= 36))
df_lane2 = df_lane2.with_columns(pl.Series("lane_id", pl.Series([2] * df_lane2.shape[0])))
df_lane3 = df.filter((pl.col('y_position') >= 36) & (pl.col('y_position') <= 48))
df_lane3 = df_lane3.with_columns(pl.Series("lane_id", pl.Series([3] * df_lane3.shape[0])))
df_lane4 = df.filter((pl.col('y_position') >= 48) & (pl.col('y_position') <= 60))
df_lane4 = df_lane4.with_columns(pl.Series("lane_id", pl.Series([4] * df_lane4.shape[0])))
# keep the columns: _id, timestamp, x_position, y_position, x_speed, lane_id and save the data frame to parquet file
df_lane1 = df_lane1.select(
    ['$oid', 'timestamp', 'x_position', 'x_speed', 'lane_id', "length", "coarse_vehicle_class"])
# change the column names to "FrameID", "LaneID", "VehID", "VehClass", "Vehlen", "LocalY", "MeanSpeed"
df_lane1.columns = ["VehID", "FrameID", "LocalY", "MeanSpeed", "LaneID", "Vehlen", "VehClass"]
df_lane1.write_parquet("./data/I24_data_lane1.parquet")
df_lane2 = df_lane2.select(
    ['$oid', 'timestamp', 'x_position', 'x_speed', 'lane_id', "length", "coarse_vehicle_class"])
df_lane2.columns = ["VehID", "FrameID", "LocalY", "MeanSpeed", "LaneID", "Vehlen", "VehClass"]
df_lane2.write_parquet("./data/I24_data_lane2.parquet")
df_lane3 = df_lane3.select(
    ['$oid', 'timestamp', 'x_position', 'x_speed', 'lane_id', "length", "coarse_vehicle_class"])
df_lane3.columns = ["VehID", "FrameID", "LocalY", "MeanSpeed", "LaneID", "Vehlen", "VehClass"]
df_lane3.write_parquet("./data/I24_data_lane3.parquet")
df_lane4 = df_lane4.select(
    ['$oid', 'timestamp', 'x_position', 'x_speed', 'lane_id', "length", "coarse_vehicle_class"])
df_lane4.columns = ["VehID", "FrameID", "LocalY", "MeanSpeed", "LaneID", "Vehlen", "VehClass"]
df_lane4.write_parquet("./data/I24_data_lane4.parquet")
print("Finish filtering the data!")
