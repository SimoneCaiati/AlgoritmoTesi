# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import os
import shutil
from sklearn.metrics import mean_absolute_error
from UsefullModules.fromMbientlab import prepare_data_MbientLab
from UsefullModules.fromSensorLogger import prepare_data_SensorLogger
from PositionGetters.PositionalDatas1 import PositionalDatas1 as pd1
from PositionGetters.PositionalDatas2 import PositionalDatas2 as pd2
from PositionGetters.PositionalDatas3 import PositionalDatas3 as pd3
from PositionGetters.PositionalDatas4 import PositionalDatas4 as pd4
from PositionGetters.PositionalDatas5 import PositionalDatas5 as pd5


def main(path, scelta):
    # in base alla scelta fatta da interfaccia salvo la cartella nell'apposita cartella del progetto
    if scelta=="Cellulare":
        directory="SensorLogger"
        Destinazione = os.path.join(f"{directory}/File_divisi", os.path.basename(path))
        shutil.copytree(path, Destinazione, dirs_exist_ok=True)
    else: 
        directory=="MbientLab"
        Destinazione = os.path.join(f"{directory}/File_divisi", os.path.basename(path))
        shutil.copytree(path, Destinazione, dirs_exist_ok=True)
    
    # nome della cartella 
    file_index=os.path.basename(path)
    sample_rate=100

    trained =  pd.read_csv(f"{directory}/Training/p_{file_index}_reconstructed.csv", delimiter=',', na_values=['']).replace(" ", "").dropna().to_numpy().astype(float)

    if directory =="MbientLab":
        DataFrame = prepare_data_MbientLab(directory, file_index)
    else:
        DataFrame = prepare_data_SensorLogger(directory, file_index)

    t_0=0
    t_n=DataFrame[len(DataFrame)-1,0]
 
    condizione = (DataFrame[:, 0] >= t_0) & (DataFrame[:, 0] <= t_n)
    datapicked = DataFrame[condizione, :]

    timestamp=datapicked[:,0]
    accelerometerData=datapicked[:,1:4]
    gyroscopeData=datapicked[:,4:7]
    orientationData=datapicked[:,7:10]
    magnetometerData=datapicked[:,10:13]
    barometerData=datapicked[:,13:15]

    # 1° BLOCCO
    p_d1= pd1(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, False)   
    p_d1.file_manager.create_directories()
    p_d1.processData() 
    differenze = np.diff(p_d1.position, axis=0)
    distanze = np.linalg.norm(differenze, axis=1)
    distanza_totale = np.sum(distanze)
    print(f"Distanza PD1:{distanza_totale}")
    # 2° BLOCCO
    p_d2= pd2(timestamp, p_d1.earthAcc, orientationData, sample_rate, file_index, directory, False)                                
    p_d2.file_manager.create_directories()
    p_d2.processData()
    differenze = np.diff(p_d2.position, axis=0)
    distanze = np.linalg.norm(differenze, axis=1)
    distanza_totale = np.sum(distanze)
    print(f"Distanza PD2:{distanza_totale}")
    # 3° BLOCCO
    p_d3= pd3(timestamp, p_d2.ifft_signal, orientationData, sample_rate, file_index, directory, False, magnetometerData)         
    p_d3.file_manager.create_directories()
    p_d3.processData()
    differenze = np.diff(p_d3.position, axis=0)
    distanze = np.linalg.norm(differenze, axis=1)
    distanza_totale = np.sum(distanze)
    print(f"Distanza PD3:{distanza_totale}")
    # 4° BLOCCO
    p_d4 = pd4(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, False, magnetometerData, gyroscopeData, barometerData, p_d1.earthAcc)
    p_d4.file_manager.create_directories()
    p_d4.processData()
    # 5° BLOCCO
    p_d5 = pd5(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, False, magnetometerData, gyroscopeData, barometerData, p_d1.earthAcc, p_d2.ifft_signal,p_d1.position,p_d3.position)
    p_d5.file_manager.create_directories()
    distanza = p_d5.processData()
    tempoImpiegato = timestamp[-1]
    velocitaMedia = np.amax(p_d1.velocity)

    return distanza, round(tempoImpiegato,2), round(velocitaMedia,2)

            