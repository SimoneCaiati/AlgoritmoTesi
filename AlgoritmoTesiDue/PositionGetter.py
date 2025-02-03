import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.signal import butter, filtfilt
from ProjectManager import DirManager
from KalmanFilter import KalmanFilter

class PositionGetter:
    
    def __init__(self,timestamp,accelerometerData,gyroscopeData,orientationData,magnetometer,sample_rate,file_index):
        self.timestamp=timestamp
        self.Acc=accelerometerData
        self.Gyro=gyroscopeData
        self.Orient=orientationData
        self.Mag=magnetometer
        self.sample_rate=sample_rate
        self.file_index=file_index
        self.directory="SensorLogger"
        
        if self.file_index != '0':                                      # '0' è il file_index dei test
            self.file_manager= DirManager(self.directory,self.file_index)

    def processData(self):
        self.getELA()
        self.identify_moving_periods()
        self.getPositionData(self.earthAcc,"PositionalData1")
        #self.fastFourierTransform()                                     
        #self.getPositionData(self.ifft_signal,"PositionalData2")  
        #self.applicateKalman()
        #self.getPositionData(self.kalman_acc,"PositionalData3") 

    # metodo che mi consente di ottenere l'accelerazione terrestre atraverso il prodotto matriacale delle matrici rotazionali (pitch,roll,yaw) e l'accelerazione lineare
    def getELA(self):
        self.earthAcc=np.empty((len(self.timestamp),3))
        self.delta_time = np.diff(self.timestamp, prepend=self.timestamp[0])
        print(f"vettore delta time:\n{self.delta_time}")
        print(f"massimo valore di delta time:{self.delta_time[1:].max()} e minimo:{self.delta_time[1:].min()}")

        for index in range(len(self.timestamp)):
            self.orientationTest=np.zeros((len(self.timestamp),3))
            # matrice pitch rotazionale 
            matA=np.array([[np.cos(self.Orient[index,0]),-np.sin(self.Orient[index,0]),0],
                            [np.sin(self.Orient[index,0]),np.cos(self.Orient[index,0]),0],
                            [0,0,1]])
            # matrice roll rotazionale
            matB=np.array([[np.cos(self.Orient[index,1]),0,np.sin(self.Orient[index,1])],
                            [0,1,0],
                            [-np.sin(self.Orient[index,1]),0,np.cos(self.Orient[index,1])]])
            # matrice yaw rotazionale
            matC=np.array([[1,0,0],
                            [0,np.cos(self.Orient[index,2]),-np.sin(self.Orient[index,2])],
                            [0,np.sin(self.Orient[index,2]),np.cos(self.Orient[index,2])]])
            
            matR = matC @ matB @ matA  # matrice rotazionale

            # Trasformazione dell'accelerazione nel sistema terrestre
            self.earthAcc[index] = matR @ self.Acc[index]
        print(f"Accelerazione terrestre:\n{self.earthAcc}")


    # METODO CHE IDENTIFICA I PERIODI DI MOVIMENTO SULLA BASE DELL'ACCELERAZIONE TOTALE COMPARATA AD UN THRESHOLD DINAMICO E AD UN MARGINE IN AVANTI ED INDIETRO
    def identify_moving_periods(self):
        self.is_moving = np.empty(len(self.timestamp))
        margin=1
        
        magnitudes = np.sqrt(np.sum(self.earthAcc**2, axis=1))  # accelerazione risultante
        print(f"Magnitudine:\n{magnitudes}")
        print(f"massimo: {np.max(magnitudes)} media: {np.mean(magnitudes)} minimo: {np.min(magnitudes)}")
        
        for index in range(len(self.timestamp)):
            self.is_moving[index] = np.sqrt(
                self.earthAcc[index].dot(self.earthAcc[index])) >= np.mean(magnitudes)-0.5  # threshold (0.5 è una tolleranza rispetto alla media)

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
        print(f"Positional data:\n{self.position}")
        PositionDataFrame=pd.DataFrame(self.position)
        self.file_manager.save_position_data(PositionDataFrame, stringa)
    
    
    # FILTRO PASSA-BASSO PER FILTRARE I DATI 
    def butter_lowpass_filter(self,data, cutoff, fs, order=2):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        filtered_data = np.zeros_like(data) 
        for i in range(data.shape[1]):  # Itera sulle colonne (assi X, Y, Z)
            filtered_data[:, i] = filtfilt(b, a, data[:, i])
        return filtered_data        


    # IMPLEMENTAZIONE DELLA FAST FOURIER TRANSFORM
    def fastFourierTransform(self):
        # Visualizzazione del segnale originale nel dominio del tempo
        plt.figure(figsize=(10, 4))
        plt.plot(self.timestamp, self.earthAcc[:, 0], label="x")
        plt.plot(self.timestamp, self.earthAcc[:, 1], label="y")
        plt.plot(self.timestamp, self.earthAcc[:, 2], label="z")
        plt.title("Accelerazione lineare terrestre (dominio del tempo)")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Accelerazione lineare terrestre")
        plt.legend()
        if self.file_index != '0':                                      # '0' è il file_index dei test
            plt.savefig(self.file_manager.fastFourierDir + "/" + "Accelerazione_lineare_terrestre.png")
        plt.show()

        # Applicazione della FFT per ciascun asse
        fft_acceleration = np.fft.fft(self.earthAcc, axis=0)
        frequencies = np.fft.fftfreq(len(self.timestamp), d=1/self.sample_rate)

        # Visualizzazione dello spettro di frequenza
        plt.figure(figsize=(10, 4))
        plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_acceleration[:len(frequencies)//2, 0]), label="x")
        plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_acceleration[:len(frequencies)//2, 1]), label="y")
        plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_acceleration[:len(frequencies)//2, 2]), label="z")
        plt.title("Spettro di frequenza dell'accelerazione")
        plt.xlabel("Frequenza (Hz)")
        plt.ylabel("Ampiezza")
        plt.legend()
        if self.file_index != '0':                                      # '0' è il file_index dei test
            plt.savefig(self.file_manager.fastFourierDir + "/" + "Spettro_frequenza_accelerazione.png")
        plt.show()

        # Parametri del filtro
        cutoff = 2.5  # Frequenza di taglio (Hz)

        # Filtro nel dominio della frequenza per ciascun asse
        fft_filtered_signal = fft_acceleration.copy()
        fft_filtered_signal[np.abs(frequencies) > cutoff, :] = 0  # Imposta a 0 le frequenze oltre il cutoff

        # Ritorno al dominio del tempo usando la IFFT
        ifft_signal = np.fft.ifft(fft_filtered_signal, axis=0)
        self.ifft_signal = np.real(ifft_signal)  # Prendi solo la parte reale (il segnale filtrato)

        # Visualizzazione del segnale filtrato (dominio del tempo)
        plt.figure(figsize=(10, 4))
        plt.plot(self.timestamp, self.earthAcc[:, 0], label="Segnale originale x", alpha=0.5)
        plt.plot(self.timestamp, self.ifft_signal[:, 0], label="Segnale filtrato x", linewidth=2)
        plt.plot(self.timestamp, self.ifft_signal[:, 1], label="Segnale filtrato y", linewidth=2)
        plt.plot(self.timestamp, self.ifft_signal[:, 2], label="Segnale filtrato z", linewidth=2)
        plt.title("Segnale filtrato con IFFT (dominio del tempo)")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Accelerazione")
        plt.legend()
        if self.file_index != '0':                                      # '0' è il file_index dei test
            plt.savefig(self.file_manager.fastFourierDir + "/" + "Segnale_Filtrato_IFFT.png")
        plt.grid()
        plt.show()

        
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
            z = np.array([self.ifft_signal[index,0],self.ifft_signal[index,1],self.ifft_signal[index,2],
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