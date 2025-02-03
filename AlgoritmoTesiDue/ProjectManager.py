import os
import pandas as pd 
import numpy as np
class DirManager:
    def __init__(self, base_dir, file_index):
        self.file_PD_Dir = os.path.join(base_dir, "PositionalDatas", str(file_index))
        self.mediaDir = os.path.join( "Media", str(file_index))
        self.fastFourierDir = os.path.join(self.mediaDir, "FastFourierPlots")
        self.create_directories()

    def create_directories(self):
        os.makedirs(self.file_PD_Dir, exist_ok=True)
        os.makedirs(self.mediaDir, exist_ok=True)
        os.makedirs(self.fastFourierDir, exist_ok=True)

    def save_position_data(self, data, filename):
        df = pd.DataFrame(data)
        df.to_csv(os.path.join(self.file_PD_Dir, filename + ".csv"), index=False)
        
def load_and_rename_csv(filepath, rename_dict, drop_cols=None, skiprows=None):
    df = pd.read_csv(filepath, delimiter=',', na_values=[''], skiprows=skiprows)
    if drop_cols:
        df = df.drop(columns=drop_cols)
    return df.rename(columns=rename_dict)

def prepare_data(directory,file_index):
    path = f"{directory}/File_uniti/{file_index}.csv"

    if os.path.exists(path)== False:    # creazione del file
        fileAccelerometer=load_and_rename_csv(f"{directory}/File_divisi/{file_index}/Accelerometer.csv",{'seconds_elapsed': 'Timestamp','x': 'Acc x (m/s^2)','y': 'Acc y (m/s^2)','z': 'Acc z (m/s^2)',},drop_cols="time")
        fileGyroscope=load_and_rename_csv(f"{directory}/File_divisi/{file_index}/Gyroscope.csv",{'seconds_elapsed': 'Timestamp','x': 'Gyro x (rad/s)','y': 'Gyro y (rad/s)','z': 'Gyro z (rad/s)',},drop_cols="time")
        fileMagnetometer=load_and_rename_csv(f"{directory}/File_divisi/{file_index}/Magnetometer.csv",{'seconds_elapsed': 'Timestamp','x': 'Mag x (microTesla)','y': 'Mag y (microTesla)','z': 'Mag z (microTesla)',},drop_cols="time")
        fileOrientation=load_and_rename_csv(f"{directory}/File_divisi/{file_index}/Orientation.csv",{'seconds_elapsed': 'Timestamp','pitch': 'Beccheggio/Pitch (rad/s)','roll': 'Rollio/Roll (rad/s)','yaw': 'Imbardata/Yaw (rad/s)'},drop_cols=["time","qx","qz","qy","qw"])
        fileBarometer=load_and_rename_csv(f"{directory}/File_divisi/{file_index}/Barometer.csv",{'seconds_elapsed': 'Timestamp','pressure': 'Pressione (milliBar)','relativeAltitude': 'Altitudine (m)',},drop_cols="time")
    
        index="Timestamp"
    
        file_t1=pd.merge(fileAccelerometer, fileGyroscope, on=index, how="inner")
        file_t2=pd.merge(file_t1, fileOrientation, on=index, how="inner")
        fileData=pd.merge(file_t2, fileMagnetometer, on=index, how="inner")
        fileData=fileData[[index,"Acc x (m/s^2)","Acc y (m/s^2)","Acc z (m/s^2)","Gyro x (rad/s)","Gyro y (rad/s)","Gyro z (rad/s)","Beccheggio/Pitch (rad/s)","Rollio/Roll (rad/s)","Imbardata/Yaw (rad/s)","Mag x (microTesla)","Mag y (microTesla)","Mag z (microTesla)"]]
    
        # Interpolazione del barometro rispetto al timestamp degli altri sensori
        def interpola_barometro(data_df, baro_df):
            timestamps = data_df['Timestamp'].to_numpy()
            altitudes = np.interp(timestamps, baro_df['Timestamp'].to_numpy(), baro_df['Altitudine (m)'].to_numpy())
            pressures = np.interp(timestamps, baro_df['Timestamp'].to_numpy(), baro_df['Pressione (milliBar)'].to_numpy())
    
            data_df['Altitudine (m)'] = altitudes
            data_df['Pressione (milliBar)'] = pressures
            return data_df

        # la funzione di interpolazione
        dati_unificati = interpola_barometro(fileData, fileBarometer)

        dati_unificati.to_csv(directory + "/File_uniti/" + file_index +".csv", index=False)
        
    return pd.read_csv(path, delimiter=',', na_values=['']).replace(" ", "").to_numpy().astype(float)




