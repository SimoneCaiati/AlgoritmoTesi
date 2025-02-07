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

        # Matrice di osservazione H: Identità (significa che leggiamo direttamente tutto lo stato)
        H = np.eye(9)            

        # Matrice di rumore di processo Q: definisce l'incertezza nel modello del sistema
        Q = np.eye(9) * 0.1                        

        # Matrice di rumore della misura R: definisce l'incertezza nelle osservazioni del sensore (aumentarlo -> diminuisce il rumore)
        R = np.eye(9) * 0.1  

        # Matrice di covarianza iniziale P: indica quanto ci fidiamo della nostra stima iniziale
        P = np.eye(9) * 10  

        # Stato iniziale x0: inizialmente supponiamo che tutte le variabili siano zero
        x0 = np.zeros(9)  

        # Matrice di transizione di stato F: modella l'evoluzione dello stato nel tempo
        # Qui viene usata una rappresentazione di moto con accelerazione per ogni asse
        F = np.array([
            [1, dt, 0, 0, 0, 0, 0.5 * dt**2, 0, 0],     # Posizione X dipende dalla velocità e accelerazione
            [0, 1,  0, 0, 0, 0, dt, 0, 0],              # Velocità X dipende dall'accelerazione
            [0, 0,  1, dt, 0, 0, 0, 0.5 * dt**2, 0],    # Posizione Y dipende dalla velocità e accelerazione
            [0, 0,  0, 1,  0, 0, 0, dt, 0],             # Velocità Y dipende dall'accelerazione
            [0, 0,  0, 0,  1, dt, 0, 0, 0.5 * dt**2],   # Posizione Z dipende dalla velocità e accelerazione
            [0, 0,  0, 0,  0, 1,  0, 0, dt],            # Velocità Z dipende dall'accelerazione
            [0, 0,  0, 0,  0, 0,  1, dt, 0],            # Accelerazione X rimane invariata
            [0, 0,  0, 0,  0, 0,  0, 1, dt],            # Accelerazione Y rimane invariata
            [0, 0,  0, 0,  0, 0,  0, 0, 1]              # Accelerazione Z rimane invariata
        ])

        # Aggiunta di un piccolo valore alla diagonale di F per maggiore stabilità numerica
        F += np.eye(F.shape[0]) * 1e-3  

        # Matrice di controllo B: viene inizializzata a zero, ma con una struttura specifica
        B = np.zeros((9, 9))
        np.fill_diagonal(B[:6, :6], 1)  # Assegna 1 alle prime 6 diagonali per gestire la posizione e velocità
        np.fill_diagonal(B[3:, 3:], 1)  # Assegna 1 agli ultimi 6 per gestire velocità e accelerazione

        # Creazione del filtro di Kalman con i parametri definiti
        kf = KalmanFilter(F, B, H, Q, R, P, x0)

        # Lista per memorizzare le predizioni del filtro di Kalman
        predictions = []

        # Iteriamo su tutti i campioni dei dati
        for index in range(num_samples):
            # Combiniamo le misure dei tre sensori (Accelerometro, Orientamento, Magnetometro)
            z = np.concatenate((self.Acc[index], self.Orient[index], self.Mag[index]))

            # Effettuiamo la predizione con il filtro di Kalman e salviamo la stima della posizione
            predictions.append(kf.predict()[0:3])

            # Aggiorniamo il filtro con la nuova misura
            kf.update(z)

            # Salviamo i risultati filtrati per accelerometro, orientamento e magnetometro
            self.kalman_acc[index] = kf.get_state()[0:3]        # Prima parte dello stato = accelerazione filtrata
            self.kalman_orient[index] = kf.get_state()[3:6]     # Seconda parte dello stato = orientamento filtrato
            self.kalman_mag[index] = kf.get_state()[6:9]        # Terza parte dello stato = magnetometro filtrato


        # import matplotlib.pyplot as plt

        # print("Varianza originale Acc:", np.var(self.Acc, axis=0))
        # print("Varianza filtrata Acc:", np.var(self.kalman_acc, axis=0))
        # print("Predizione:", kf.predict().T)
        # print("Misura:", z.T)
        # print("Errore di innovazione (y):", (z - np.dot(kf.H, kf.get_state())).T)
        # print("Guadagno di Kalman (K):", kf.K)
        # print("Covarianza P:", kf.P)

        # plt.plot(range(len(self.Acc)), self.Acc, label='Measurements')
        # plt.plot(range(len(predictions)), np.array(predictions), label='Kalman Filter Prediction')
        # plt.legend()
        # plt.show()
        

        

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

    def predict(self, u=np.zeros(9)):
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        self.y = z - np.dot(self.H, self.x)  
        S = self.R + np.dot(np.dot(self.H, self.P), self.H.T)  
    
        if np.linalg.det(S) == 0:
            S += np.eye(S.shape[0]) * 1e-3  # Maggiore stabilità numerica
    
        self.K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  

        self.x = self.x + np.dot(self.K, self.y)

        self.P = (self.I - np.dot(self.K, self.H)) @ self.P + np.eye(self.n) * 0.005  # Maggiore stabilità

        self.P = (self.P + self.P.T) / 2  

    def get_state(self):
        return self.x
