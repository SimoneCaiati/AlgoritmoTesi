# -*- coding: utf-8 -*-
from UsefullModules.fromMbientlab import prepare_data_MbientLab
from UsefullModules.fromSensorLogger import prepare_data_SensorLogger
from PositionGetters.PositionalDatas1 import PositionalDatas1 as pd1
from PositionGetters.PositionalDatas2 import PositionalDatas2 as pd2
from PositionGetters.PositionalDatas3 import PositionalDatas3 as pd3
from PositionGetters.PositionalDatas4 import PositionalDatas4 as pd4
from PositionGetters.PositionalDatas5 import PositionalDatas5 as pd5

# Dati che possono variare: directory, file_index, sample_rate
directory="SensorLogger"
file_index="9metri_dritto_destra"
sample_rate=100

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
# 2° BLOCCO
p_d2= pd2(timestamp, p_d1.earthAcc, orientationData, sample_rate, file_index, directory, False)                                
p_d2.file_manager.create_directories()
p_d2.processData()
# 3° BLOCCO
p_d3= pd3(timestamp, p_d2.ifft_signal, orientationData, sample_rate, file_index, directory, False, magnetometerData)         
p_d3.file_manager.create_directories()
p_d3.processData()
# 4° BLOCCO
p_d4 = pd4(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, False, magnetometerData, gyroscopeData, barometerData, p_d1.earthAcc)
p_d4.file_manager.create_directories()
p_d4.processData()
# 5° BLOCCO
p_d5 = pd5(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, False, magnetometerData, gyroscopeData, barometerData, p_d1.earthAcc, p_d2.ifft_signal,p_d1.position,p_d3.position)
p_d5.file_manager.create_directories()
p_d5.processData()