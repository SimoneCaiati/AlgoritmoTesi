from hmac import new
import numpy as np
import pandas as pd 
import os
from acceleration_path import AccPath

def main():
    directory="MbientLab"
    file_index="ADL1"
    path= directory +"/" + "File_uniti/" + file_index + ".csv"

    if os.path.exists(path)== False:
        fileAccelerometer=(pd.read_csv(directory +"/File_divisi/"+ file_index + "/" + "Accelerometer.csv",delimiter=',',na_values=[''])).drop(columns=["epoc (ms)","timestamp (+0100)"])
        fileGyroscope=(pd.read_csv(directory +"/File_divisi/"+ file_index + "/" + "Gyroscope.csv",delimiter=',', na_values=[''])).drop(columns=["epoc (ms)","timestamp (+0100)"])
    
        new_column_gravity={
        'elapsed (s)': 'Timestamp',
        'x-axis (g)': 'Gx',
        'y-axis (g)': 'Gy',
        'z-axis (g)': 'Gz',
        }
        new_column_gyroscope={
        'elapsed (s)': 'Timestamp',
        'x-axis (deg/s)': 'Wx',
        'y-axis (deg/s)': 'Wy',
        'z-axis (deg/s)': 'Wz',
        }
    
        fileAccelerometer=fileAccelerometer.rename(columns=new_column_gravity)
        fileGyroscope=fileGyroscope.rename(columns=new_column_gyroscope)
    
        column="Timestamp"
        file=pd.merge(fileAccelerometer, fileGyroscope, on=column, how="inner")
        file=file[["Timestamp","Wx","Wy","Wz","Gx","Gy","Gz"]]
        file.to_csv(directory + "/File_uniti/" + file_index +".csv", index=False)
 
    file=pd.read_csv(directory +"/"+ "File_uniti/" + file_index + ".csv",delimiter=',', skiprows=1, na_values=['']) 
    data = file.replace(" ", "").to_numpy().astype(float)

    t_0=0
    t_n=data[len(data)-1,0]
 
    condizione = (data[:, 0] >= t_0) & (data[:, 0] <= t_n)
    datapicked = data[condizione, :]
    timestamp=datapicked[:,0]

    accelerometerData=datapicked[:,4:7]
    gyroscopeData=datapicked[:,1:4]

    #magnetometerData=datapicked[:,7:10]

    acc_path = AccPath(
            gyroscopeData, 
            accelerometerData, 
            timestamp,
            sample_rate=100, 
            saving_path="media/"+file_index+"/")

    acc_path.percorso()
    #acc_path.animate_path(length_sec=15, fps=4)
    acc_path.plot_sensor_data()
    #acc_path.plot_euler_angles()
    acc_path.plot_internal_states()
    acc_path.plot_acceleration_velocity_position()
    acc_path.print_distance_start_final()

    PositionDataFrame=pd.DataFrame(acc_path.position)                     
    PositionDataFrame.to_csv(os.path.join("media/"+file_index+"/", "position.csv"), index=False)