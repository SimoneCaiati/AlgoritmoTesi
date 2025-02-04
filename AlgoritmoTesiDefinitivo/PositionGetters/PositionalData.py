from cgi import test
import numpy as np
import pandas as pd 

from UsefullModules.ProjectManager import DirManager
from UsefullModules.Visualizer import Visualizer

class PositionalData:
    
    def __init__(self,timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test,specificPD):
        self.timestamp=timestamp
        self.Acc=accelerometerData
        self.Orient=orientationData
        self.sample_rate=sample_rate
        self.file_index=file_index
        self.directory=directory
        self.position=None
        self.delta_time = np.diff(self.timestamp, prepend=self.timestamp[0])
        
        self.file_manager= DirManager(self.directory, self.file_index, specificPD, test)
        self.visualizer=Visualizer(self.position, self.timestamp, self.file_manager.mediaDir)


    # METODO CHE IDENTIFICA I PERIODI DI MOVIMENTO SULLA BASE DELL'ACCELERAZIONE TOTALE COMPARATA AD UN THRESHOLD DINAMICO E AD UN MARGINE IN AVANTI ED INDIETRO
    def identify_moving_periods(self, accelerazione):
        self.is_moving = np.empty(len(self.timestamp))
        margin=1
        
        magnitudes = np.sqrt(np.sum(accelerazione**2, axis=1))  # accelerazione risultante
        print(f"Magnitudine:\n{magnitudes}")
        print(f"massimo: {np.max(magnitudes)} media: {np.mean(magnitudes)} minimo: {np.min(magnitudes)}")
        
        for index in range(len(self.timestamp)):
            print(np.sqrt(accelerazione[index].dot(self.Acc[index])))
            self.is_moving[index] = np.sqrt(accelerazione[index].dot(accelerazione[index])) >= np.mean(magnitudes)-0.5  # threshold (0.5 è una tolleranza rispetto alla media)

        if self.sample_rate !=1:
            margin = int(0.1 * self.sample_rate)  

        for index in range(len(self.timestamp) - margin):
            self.is_moving[index] = any(self.is_moving[index:(index + margin)])  # add leading margin

        for index in range(len(self.timestamp) - 1, margin, -1):
            self.is_moving[index] = any(self.is_moving[(index - margin):index])  # add trailing margin
        print(f"movimento per timesample:\n{self.is_moving}")
    

    # METODO CHE SULLA BASE DELL'ACCELERAZIONE RICEVUTA IN INPUT E LA STRINGA CORRISPONDENTE GENERA I PD1, PD2 O PD3
    def getPositionData(self,accelerazione,stringa):
        self.velocity = np.zeros((len(self.timestamp), 3))
        self.position = np.zeros((len(self.timestamp), 3)) 
        
        for index in range(len(self.timestamp)): # solo se in movimento trovo la velocità
            if self.is_moving[index]:
                self.velocity[index]= self.velocity[index-1] + self.delta_time[index]* accelerazione[index]
                
        print(f"velocita:\n{self.velocity}")
               
        for index in range(len(self.timestamp)):
            if self.is_moving[index]:
                self.position[index] =self.delta_time[index] * self.velocity[index] + (0.5*accelerazione[index]*(self.delta_time[index]**2)) # nella legge oraria andrebbe anche considerata la posizione iniziale 
            else:
                self.position[index]=self.position[index-1]
        print(f"Positional data:\n{self.position}")
        PositionDataFrame=pd.DataFrame(self.position)
        self.visualizer.position=self.position.copy()                           # sovrascrivo il vettore posizione del visualizer, che è impostato a None di default 
        if not test:
            self.file_manager.save_position_data(PositionDataFrame, stringa)

    def plotGraphics(self, nome_acc, nome_orient ,Acc, Orient):
        self.visualizer.plot_path()
        self.visualizer.plot_acceleration_data(nome_acc, Acc)
        self.visualizer.plot_euler_angles(nome_orient, Orient)
    

