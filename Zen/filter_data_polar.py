import polars as pl

# use polars to read the csv file D:\ZEN\ROUTE11\L001_F005_ALL\L001_F005_TRAJECTORY\L001_F005_trajectory.csv
# schema:vehicle_id, datatime, vehicle_type, velocity, traffic_lane, longitude, latitude, kilopost, length, detected_flag
data = pl.read_csv("D:/ZEN/ROUTE11/L001_F005_ALL/L001_F005_TRAJECTORY/L001_F005_trajectory.csv", has_header=False,
                   new_columns=['vehicle_id', 'datatime', 'vehicle_type', 'velocity', 'traffic_lane', 'longitude',
                                'latitude',
                                'kilopost', 'length', 'detected_flag'])

# vehicle_type – 1: normal vehicle 2: large vehicle(bus, truck, etc.)
# datatime - HHMMSSFFF in string; kilopost - Distance from starting point of the expressway route in meter
# lane - 1: driving lane 2: passing lane 3: entrance lane

############ First round: filter the data with kilopost in the range of [3500, 4200] and in the passing lane, filter out the datetime < 100000000
data = data.filter(
    (pl.col('kilopost') >= 3000) & (pl.col('kilopost') <= 4200) & (pl.col('traffic_lane') == 2) & (pl.col(
        'datatime') >= 100000000))
print("The number of vehicles in the dataset is: ", data['vehicle_id'].n_unique(), "and ", data.shape[0], " points.")
# change the unit of the velocity from km/h to m/s
data = data.with_columns((pl.col('velocity') / 3.6).alias('velocity_m_s'))

# convert the FrameID to string
data = data.with_columns(string_frame=pl.col('datatime').cast(pl.Utf8))
# convert string "HHMMSSFFF" to datetime
data = data.with_columns(time=pl.col('string_frame').str.to_datetime("%H%M%S%3f"))
# make a new column named Epoch for epoch time
data = data.with_columns(epoch=pl.col('time').dt.epoch(time_unit='ms'))
df = data.select(['vehicle_id', 'datatime', 'vehicle_type', 'velocity_m_s', 'traffic_lane', "length", "kilopost", "epoch", "time"])
df.columns = ["VehID", "FrameID", "VehClass", "MeanSpeed", "LaneID", "Vehlen", "LocalY", "Epoch", "time"]
df.write_parquet("./data/Zen_data.parquet")
print("Finish filtering the data!")
