import numpy as np
from PositionGetters.PositionalData import PositionalData 

class PositionalDatas3(PositionalData):
    def __init__(self,timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test, magnetometerData):
        super().__init__(timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test,specificPD="PD3") 
        self.Mag=magnetometerData

    def processData(self):
        self.identify_moving_periods(self.Acc)
        self.applicateKalman()
        self.getPositionData(self.kalman_acc,"PositionalData3")
        self.plotGraphics("Accelerazione_after_Kalman","Angoli_after_Kalman",self.kalman_acc,self.kalman_orient)
        
    def applicateKalman(self):
        num_samples = len(self.timestamp)
    
        self.kalman_acc = np.empty((num_samples, 3))
        self.kalman_orient = np.empty((num_samples, 3))
        self.kalman_mag = np.empty((num_samples, 3))

        dt = 1 / self.sample_rate  

        H = np.eye(9)  
        Q = np.eye(9) * 0.1  # Aumentato per maggiore reattività
        R = np.eye(9) * 0.01  # Ridotto per dare più peso alle osservazioni
        P = np.eye(9) * 10  # Inizializzato più alto per consentire adattabilità
        x0 = np.zeros(9)  

        F = np.array([
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
        
        F += np.eye(F.shape[0]) * 1e-3  # Aggiunto smorzamento per stabilità
    
        B = np.zeros((9, 9))
        np.fill_diagonal(B[:6, :6], 1)
        np.fill_diagonal(B[3:, 3:], 1)

        kf = KalmanFilter(F, B, H, Q, R, P, x0)

        for index in range(num_samples):
            z = np.concatenate((self.Acc[index], self.Orient[index], self.Mag[index]))
            kf.predict()
            kf.update(z)
            #print(kf.get_state())
            self.kalman_acc[index] = kf.get_state()[0,0:3]
            self.kalman_orient[index] = kf.get_state()[0,3:6]
            self.kalman_mag[index] = kf.get_state()[0,6:9]

        print("Varianza originale Acc:", np.var(self.Acc, axis=0))
        print("Varianza filtrata Acc:", np.var(self.kalman_acc, axis=0))

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

    def predict(self, u=0):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        y = z - np.dot(self.H, self.x)  
        S = self.R + np.dot(np.dot(self.H, self.P), self.H.T)  
    
        if np.linalg.det(S) == 0:
            S += np.eye(S.shape[0]) * 1e-3  # Maggiore stabilità numerica
    
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  

        self.x = self.x + np.dot(K, y)

        self.P = (self.I - np.dot(K, self.H)) @ self.P + np.eye(self.n) * 0.005  # Maggiore stabilità

        self.P = (self.P + self.P.T) / 2  

    def get_state(self):
        return self.x
