import numpy as np
import pytest
from PositionGetters.PositionalDatas1 import PositionalDatas1 as pd1
from PositionGetters.PositionalDatas2 import PositionalDatas2 as pd2
from PositionGetters.PositionalDatas3 import PositionalDatas3 as pd3
from PositionGetters.PositionalDatas4 import PositionalDatas4 as pd4
from PositionGetters.PositionalDatas5 import PositionalDatas5 as pd5

from UsefullModules.fromSensorLogger import prepare_data_SensorLogger

@pytest.fixture
def sample_data():
    """Fixture per creare dati di input di esempio"""
    directory="SensorLogger"
    file_index="Test_moto_rettilineo_acc_ideale"
    sample_rate=100

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

    return timestamp,accelerometerData,gyroscopeData,orientationData,magnetometerData,barometerData, sample_rate, file_index, directory

def test_positionalOneAndTwo(sample_data):
    timestamp,accelerometerData,gyroscopeData,orientationData,magnetometerData,barometerData, sample_rate, file_index, directory = sample_data
    
    # 1° BLOCCO
    p_d1= pd1(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, True)   
    p_d1.file_manager.create_directories()
    p_d1.processData()  
    # 2° BLOCCO
    p_d2= pd2(timestamp, p_d1.earthAcc, orientationData, sample_rate, file_index, directory, True)                                
    p_d2.file_manager.create_directories()
    p_d2.processData()
    
    assert accelerometerData.shape == p_d1.earthAcc.shape
    assert accelerometerData.shape == p_d2.ifft_signal.shape
    p_d1.visualizer.plot_path()
    p_d2.visualizer.plot_path()
    
def test_positionalOneAndThree(sample_data):
    timestamp,accelerometerData,gyroscopeData,orientationData,magnetometerData,barometerData, sample_rate, file_index, directory = sample_data
    
    # 1° BLOCCO
    p_d1= pd1(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, True)   
    p_d1.file_manager.create_directories()
    p_d1.processData()  
    # 3° BLOCCO
    p_d3= pd3(timestamp, p_d1.earthAcc, orientationData, sample_rate, file_index, directory, True, magnetometerData)         
    p_d3.file_manager.create_directories()
    p_d3.processData()
    
    assert accelerometerData.shape == p_d1.earthAcc.shape
    assert p_d3.kalman_acc.shape == p_d1.earthAcc.shape
    assert p_d3.kalman_mag.shape == magnetometerData
    assert p_d3.kalman_orient.shape == orientationData
    p_d1.visualizer.plot_path()
    p_d3.visualizer.plot_path()
    
def test_positionalTwoAndThree(sample_data):
    timestamp,accelerometerData,gyroscopeData,orientationData,magnetometerData,barometerData, sample_rate, file_index, directory = sample_data

    # 2° BLOCCO
    p_d2= pd2(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, True)                                
    p_d2.file_manager.create_directories()
    p_d2.processData()
    # 3° BLOCCO
    p_d3= pd3(timestamp, p_d2.ifft_signal, orientationData, sample_rate, file_index, directory, True, magnetometerData)         
    p_d3.file_manager.create_directories()
    p_d3.processData()
    
    assert p_d3.kalman_acc.shape == p_d2.ifft_signal.shape
    assert p_d3.kalman_mag.shape == magnetometerData
    assert p_d3.kalman_orient.shape == orientationData
    p_d3.visualizer.plot_path()
    p_d2.visualizer.plot_path()
    
def test_positionalOneAndFour(sample_data):
    timestamp,accelerometerData,gyroscopeData,orientationData,magnetometerData,barometerData, sample_rate, file_index, directory = sample_data

    # 1° BLOCCO
    p_d1= pd1(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, True)   
    p_d1.file_manager.create_directories()
    p_d1.processData()  
    # 4° BLOCCO
    p_d4 = pd4(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, True, magnetometerData, gyroscopeData, barometerData, p_d1.earthAcc)
    p_d4.file_manager.create_directories()
    p_d4.test_path=["SensorLogger/Training/p_test.csv"]
    p_d4.processData()