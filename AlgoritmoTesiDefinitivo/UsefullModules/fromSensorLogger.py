import os
import pandas as pd 
import numpy as np
from UsefullModules.ProjectManager import load_and_rename_csv


def prepare_data_SensorLogger(directory, file_index):
    path = f"{directory}/File_uniti/{file_index}.csv"

    if not os.path.exists(path):  # Creazione del file se non esiste
        fileAccelerometer = load_and_rename_csv(
            f"{directory}/File_divisi/{file_index}/Accelerometer.csv",
            {'seconds_elapsed': 'Timestamp', 'x': 'Acc x (m/s^2)', 'y': 'Acc y (m/s^2)', 'z': 'Acc z (m/s^2)'},
            drop_cols="time"
        )
        fileGyroscope = load_and_rename_csv(
            f"{directory}/File_divisi/{file_index}/Gyroscope.csv",
            {'seconds_elapsed': 'Timestamp', 'x': 'Gyro x (rad/s)', 'y': 'Gyro y (rad/s)', 'z': 'Gyro z (rad/s)'},
            drop_cols="time"
        )
        fileMagnetometer = load_and_rename_csv(
            f"{directory}/File_divisi/{file_index}/Magnetometer.csv",
            {'seconds_elapsed': 'Timestamp', 'x': 'Mag x (microTesla)', 'y': 'Mag y (microTesla)', 'z': 'Mag z (microTesla)'},
            drop_cols="time"
        )
        fileOrientation = load_and_rename_csv(
            f"{directory}/File_divisi/{file_index}/Orientation.csv",
            {'seconds_elapsed': 'Timestamp', 'pitch': 'Beccheggio/Pitch (rad/s)', 'roll': 'Rollio/Roll (rad/s)', 'yaw': 'Imbardata/Yaw (rad/s)'},
            drop_cols=["time", "qx", "qz", "qy", "qw"]
        )
        fileBarometer = load_and_rename_csv(
            f"{directory}/File_divisi/{file_index}/Barometer.csv",
            {'seconds_elapsed': 'Timestamp', 'pressure': 'Pressione (milliBar)', 'relativeAltitude': 'Altitudine (m)'},
            drop_cols="time"
        )

        index = "Timestamp"

        # Merge sequenziale dei file
        file_t1 = pd.merge(fileAccelerometer, fileGyroscope, on=index, how="outer")
        file_t2 = pd.merge(file_t1, fileOrientation, on=index, how="outer")
        fileData = pd.merge(file_t2, fileMagnetometer, on=index, how="outer")

        # Riempie eventuali valori NaN con 0 per mantenere coerenza
        fileData = fileData.fillna(0)

        # Riorganizzazione delle colonne
        fileData = fileData[[index, "Acc x (m/s^2)", "Acc y (m/s^2)", "Acc z (m/s^2)",
                             "Gyro x (rad/s)", "Gyro y (rad/s)", "Gyro z (rad/s)",
                             "Beccheggio/Pitch (rad/s)", "Rollio/Roll (rad/s)", "Imbardata/Yaw (rad/s)",
                             "Mag x (microTesla)", "Mag y (microTesla)", "Mag z (microTesla)"]]

        # Funzione per interpolare il barometro
        def interpola_barometro(data_df, baro_df):
            if baro_df.empty:
                data_df['Altitudine (m)'] = 0
                data_df['Pressione (milliBar)'] = 0
            else:
                timestamps = data_df['Timestamp'].to_numpy()
                altitudes = np.interp(timestamps, baro_df['Timestamp'].to_numpy(), baro_df['Altitudine (m)'].to_numpy())
                pressures = np.interp(timestamps, baro_df['Timestamp'].to_numpy(), baro_df['Pressione (milliBar)'].to_numpy())
                data_df['Altitudine (m)'] = altitudes
                data_df['Pressione (milliBar)'] = pressures
            return data_df

        # Interpolazione del barometro
        dati_unificati = interpola_barometro(fileData, fileBarometer)

        # Salvataggio del file finale
        dati_unificati.to_csv(path, index=False)

    return pd.read_csv(path, delimiter=',', na_values=['']).replace(" ", "").to_numpy().astype(float)

