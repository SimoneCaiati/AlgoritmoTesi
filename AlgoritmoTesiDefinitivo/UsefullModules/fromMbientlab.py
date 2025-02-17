import os
import pandas as pd 
import numpy as np
from UsefullModules.ProjectManager import load_and_rename_csv


def prepare_data_MbientLab(directory, file_index):
    path = f"{directory}/File_uniti/{file_index}.csv"

    if not os.path.exists(path):  # Creazione del file se non esiste
        fileAccelerometer = load_and_rename_csv(
            f"{directory}/File_divisi/{file_index}/Accelerometer.csv",
            {'elapsed (s)': 'Timestamp', 'x-axis (g)': 'Acc x (g)', 'y-axis (g)': 'Acc y (g)', 'z-axis (g)': 'Acc z (g)'},
            drop_cols=["epoc (ms)","timestamp (+0100)"]
        )
        fileGyroscope = load_and_rename_csv(
            f"{directory}/File_divisi/{file_index}/Gyroscope.csv",
            {'elapsed (s)': 'Timestamp', 'x-axis (deg/s)': 'Gyro x (deg/s)', 'y-axis (deg/s)': 'Gyro y (deg/s)', 'z-axis (deg/s)': 'Gyro z (deg/s)'},
            drop_cols=["epoc (ms)","timestamp (+0100)"]
        )
        fileMagnetometer = load_and_rename_csv(
            f"{directory}/File_divisi/{file_index}/Magnetometer.csv",
            {'elapsed (s)': 'Timestamp', 'x': 'Mag x (microTesla)', 'y': 'Mag y (microTesla)', 'z': 'Mag z (microTesla)'},
            drop_cols="time"
        )
        fileOrientation = load_and_rename_csv(
            f"{directory}/File_divisi/{file_index}/Orientation.csv",
            {'elapsed (s)': 'Timestamp', 'pitch': 'Beccheggio/Pitch (deg/s)', 'roll': 'Rollio/Roll (deg/s)', 'yaw': 'Imbardata/Yaw (deg/s)'},
            drop_cols=["time", "qx", "qz", "qy", "qw"]
        )
        fileBarometer = load_and_rename_csv(
            f"{directory}/File_divisi/{file_index}/Barometer.csv",
            {'elapsed (s)': 'Timestamp', 'pressure': 'Pressione (milliBar)', 'relativeAltitude': 'Altitudine (m)'},
            drop_cols="time"
        )

        index = "Timestamp"

        # Merge sequenziale dei file
        file_t1 = pd.merge(fileAccelerometer, fileGyroscope, on=index, how="outer")
        file_t2 = pd.merge(file_t1, fileOrientation, on=index, how="outer")
        file_t3 = pd.merge(file_t2, fileBarometer, on=index, how="outer")
        fileData = pd.merge(file_t3, fileMagnetometer, on=index, how="outer")

        # Riempie eventuali valori NaN con 0 per mantenere coerenza
        fileData = fileData.fillna(0)

        # Riorganizzazione delle colonne
        fileData = fileData[[index, "Acc x (g)", "Acc y (g)", "Acc z (g)",
                             "Gyro x (deg/s)", "Gyro y (deg/s)", "Gyro z (deg/s)",
                             "Beccheggio/Pitch (deg/s)", "Rollio/Roll (deg/s)", "Imbardata/Yaw (deg/s)",
                             "Mag x (microTesla)", "Mag y (microTesla)", "Mag z (microTesla)","Pressione (milliBar)","Altitudine (m)"]]

        # Salvataggio del file finale
        fileData.to_csv(path, index=False)

    return pd.read_csv(path, delimiter=',', na_values=['']).replace(" ", "").to_numpy().astype(float)
