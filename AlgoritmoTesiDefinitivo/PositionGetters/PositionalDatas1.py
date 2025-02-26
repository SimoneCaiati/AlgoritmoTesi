import numpy as np

from PositionGetters.PositionalData import PositionalData 

class PositionalDatas1(PositionalData):
    
    def __init__(self,timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test):
        super().__init__(timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test,specificPD="PD1")

    def processData(self):
        self.getELA()
        self.identify_moving_periods(self.earthAcc)
        self.getPositionData(self.earthAcc,"PositionalData1")
        #self.plotGraphics("Accelerazione_terrestre","Angoli_di_Eulero",self.earthAcc,self.Orient)

    # metodo che mi consente di ottenere l'accelerazione terrestre atraverso il prodotto matriacale delle matrici rotazionali (pitch,roll,yaw) e l'accelerazione lineare
    def getELA(self):
        self.earthAcc=np.empty((len(self.timestamp),3))
        #print(f"vettore delta time:\n{self.delta_time}")
        #print(f"massimo valore di delta time:{self.delta_time[1:].max()} e minimo:{self.delta_time[1:].min()}")
        #print(f"Accelerazione prima:\n{self.Acc}")
        for index in range(len(self.timestamp)):
            # matrice pitch rotazionale 
            matX = np.array([[1, 0, 0],
                [0, np.cos(self.Orient[index,0]), np.sin(self.Orient[index,0])], 
                [0, -np.sin(self.Orient[index,0]), np.cos(self.Orient[index,0])]])

            # matrice roll rotazionale
            matY=np.array([[np.cos(self.Orient[index,1]), 0, np.sin(self.Orient[index,1])],
                [0, 1, 0],
                [-np.sin(self.Orient[index,1]), 0, np.cos(self.Orient[index,1])]])
            # matrice yaw rotazionale
            matZ=np.array([[np.cos(self.Orient[index,2]), -np.sin(self.Orient[index,2]), 0],
                [np.sin(self.Orient[index,2]), np.cos(self.Orient[index,2]), 0],
                [0, 0, 1]])
            
            matR =matZ @ matX @ matY          # matrice rotazionale
            matR = np.round(matR, decimals=10)  # Arrotondamento per evitare errori numerici
            #print(matR)

            # Trasformazione dell'accelerazione nel sistema terrestre
            self.earthAcc[index] = matR @ self.Acc[index]
        #print(f"Accelerazione terrestre dopo:\n{self.earthAcc}")


