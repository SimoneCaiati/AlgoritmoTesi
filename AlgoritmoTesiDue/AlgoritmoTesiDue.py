from PositionGetter import PositionGetter as pgot
from Visualizer import Visualizer
from ProjectManager import prepare_data

directory="SensorLogger"
test=True                                  # IMPOSTARE A TRUE PER FILE DI TEST
file_index="Test_Data_withoutOrientation"
sample_rate=100

DataFrame=prepare_data(directory,file_index)

t_0=0
t_n=DataFrame[len(DataFrame)-1,0]
 
condizione = (DataFrame[:, 0] >= t_0) & (DataFrame[:, 0] <= t_n)
datapicked = DataFrame[condizione, :]

if test:
    timestamp=datapicked[:,0]                   
    accelerometerData=datapicked[:,1:4]         
    gyroscopeData=datapicked[:,7:10]            
    orientationData=datapicked[:,4:7]           
    magnetometerData=datapicked[:,12:15]        
    barometerData=datapicked[:,10:12]           
else:
    timestamp=datapicked[:,0]
    accelerometerData=datapicked[:,1:4]
    gyroscopeData=datapicked[:,4:7]
    orientationData=datapicked[:,7:10]
    magnetometerData=datapicked[:,10:13]
    barometerData=datapicked[:,13:15]

pt= pgot(timestamp, accelerometerData, gyroscopeData, orientationData, magnetometerData, sample_rate, file_index)
pt.processData()
visualizer =Visualizer(pt.position,pt.timestamp,pt.file_manager.mediaDir)

visualizer.plot_path()
visualizer.plot_acceleration_data(accelerometerData,"Accelerometro_Iniziale")
visualizer.plot_acceleration_data(pt.ifft_signal,"Accelerometro_IFFT")
visualizer.plot_acceleration_data(pt.kalman_acc,"Accelerometro_kalman")
visualizer.plot_euler_angles(orientationData)
