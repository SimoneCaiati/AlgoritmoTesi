import numpy as np
from PositionGetters.PositionalData import PositionalData 

class PositionalDatas3(PositionalData):
    def __init__(self,timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test, magnetometerData):
        super().__init__(timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test,specificPD="PD3") 
        self.Mag=magnetometerData

    def processData(self):
        self.identify_moving_periods(self.Acc)
        self.applicateKalman()
        self.getPositionData(self.Acc,"PositionalData3")
        self.plotGraphics("Accelerazione_after_Kalman","Angoli_after_Kalman",self.kalman_acc,self.kalman_orient)

    def applicateKalman(self):
        self.kalman_acc=np.empty((len(self.timestamp),3))
        self.kalman_orient=np.empty((len(self.timestamp),3))
        self.kalman_mag=np.empty((len(self.timestamp),3))

        dt=1/self.sample_rate
        
        # Parametri iniziali del filtro di Kalman
        
        H = np.eye(9)                                   # Matrice di osservazione
        Q = np.eye(9)                                   # Rumore di processo * 0.01
        R = np.eye(9)                                   # Rumore di osservazione * 0.1  
        P = np.eye(9)                                   # Covarianza iniziale
        x0 = np.zeros(9)                                # Stato iniziale

        # Matrice F
        F = np.array([                                  # Matrice di transizione
            [1, dt, 0, 0, 0, 0, 0.5 * dt**2, 0, 0],
            [0, 1,  0, 0, 0, 0, dt, 0, 0],
            [0, 0,  1, dt, 0, 0, 0, 0.5 * dt**2, 0],
            [0, 0,  0, 1,  0, 0, 0, dt, 0],
            [0, 0,  0, 0,  1, dt, 0, 0, 0.5 * dt**2],
            [0, 0,  0, 0,  0, 1,  0, 0, dt],
            [0, 0,  0, 0,  0, 0,  1, dt, 0],
            [0, 0,  0, 0,  0, 0,  0, 1, dt],
            [0, 0,  0, 0,  0, 0,  0, 0, 1]
        ])
        # Crea una matrice 9x9 di zeri
        # Riempi la diagonale principale dei primi 6 elementi con 1
        # Riempi la diagonale principale a partire dalla quarta riga con 1
        B = np.zeros((9, 9))                            # Matrice di controllo           
        np.fill_diagonal(B[:6, :6], 1)                  
        np.fill_diagonal(B[3:, 3:], 1)                  

        kf = KalmanFilter(F, B, H, Q, R, P, x0)
        
        for index in range(len(self.timestamp)):
            
            # Costruzione del vettore di osservazione z con dati già filtrati
            z = np.array([self.Acc[index,0],self.Acc[index,1],self.Acc[index,2],
                          self.Orient[index,0],self.Orient[index,1],self.Orient[index,2],
                          self.Mag[index,0],self.Mag[index,1],self.Mag[index,2]])
    
            # Previsione del filtro di Kalman
            kf.predict(z)
    
            # Aggiornamento del filtro di Kalman con i dati osservati
            kf.update(z)
    
            # Ricostruzione cinematiche
            self.kalman_acc[index] = kf.get_state()[0:3]         
            self.kalman_orient[index]=kf.get_state()[3:6]       
            self.kalman_mag[index]= kf.get_state()[6:9]


class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0
        self.I = np.eye(self.n)

    def predict(self, u):
        """Step di predizione"""
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        """"Step di aggiornamento"""
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(np.dot(self.H,self.P) , self.H.T)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(self.R, y)
        self.P = np.dot(self.I - np.dot(K, self.H), self.P)
        	
    def get_state(self):
        """Ritorna lo stato corrente."""
        return self.x

