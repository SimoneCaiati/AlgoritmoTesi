import numpy as np
import matplotlib.pyplot as plt 
from scipy.signal import butter, filtfilt

from PositionGetters.PositionalData import PositionalData 

class PositionalDatas2(PositionalData):
    
    def __init__(self,timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test):
        super().__init__(timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test,specificPD="PD2")
        
    def processData(self):
        self.identify_moving_periods(self.Acc)
        self.fastFourierTransform()
        self.getPositionData(self.ifft_signal,"PositionalData2")
        self.plotGraphics("Accelerazione_after_IFFT","Angoli_di_Eulero",self.ifft_signal,self.Orient)

    # FILTRO PASSA-BASSO PER FILTRARE I DATI 
    def butter_lowpass_filter(self,data, cutoff, fs, order=4):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        
        filtered_data = np.zeros_like(data) 
        for i in range(data.shape[1]):  # Itera sulle colonne (assi X, Y, Z)
            filtered_data[:, i] = filtfilt(b, a, data[:, i])
        return filtered_data        
    

    # IMPLEMENTAZIONE DELLA FAST FOURIER TRANSFORM
    def fastFourierTransform(self):
        # Parametri del filtro
        cutoff = 2.5  # Frequenza di taglio (Hz)
    
        # Applicazione del filtro passa-basso di Butterworth prima della FFT
        self.filtered_Acc = self.butter_lowpass_filter(self.Acc, cutoff, self.sample_rate)

        # Visualizzazione del segnale originale e filtrato
        plt.figure(figsize=(10, 4))
        plt.plot(self.timestamp, self.Acc[:, 0], label="Originale x", alpha=0.5)
        plt.plot(self.timestamp, self.filtered_Acc[:, 0], label="Filtrato x", linewidth=2)
        plt.plot(self.timestamp, self.filtered_Acc[:, 1], label="Filtrato y", linewidth=2)
        plt.plot(self.timestamp, self.filtered_Acc[:, 2], label="Filtrato z", linewidth=2)
        plt.title("Accelerazione filtrata con Butterworth (dominio del tempo)")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Accelerazione")
        plt.legend()
        plt.grid()
        if self.file_index != '0':
            plt.savefig(self.file_manager.fastFourierDir + "/" + "Accelerazione_filtrata_Butterworth.png")
        plt.show()

        # Applicazione della FFT al segnale filtrato
        fft_acceleration = np.fft.fft(self.filtered_Acc, axis=0)
        frequencies = np.fft.fftfreq(len(self.timestamp), d=1/self.sample_rate)

        # Visualizzazione dello spettro di frequenza con indicazione della frequenza di taglio
        plt.figure(figsize=(10, 4))
        plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_acceleration[:len(frequencies)//2, 0]), label="x")
        plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_acceleration[:len(frequencies)//2, 1]), label="y")
        plt.plot(frequencies[:len(frequencies)//2], np.abs(fft_acceleration[:len(frequencies)//2, 2]), label="z")

        # Aggiunta della linea verticale per indicare la frequenza di taglio
        plt.axvline(cutoff, color='red', linestyle='--', label="Frequenza di taglio (2.5 Hz)")

        plt.title("Spettro di frequenza dell'accelerazione filtrata")
        plt.xlabel("Frequenza (Hz)")
        plt.ylabel("Ampiezza")
        plt.legend()
        plt.grid()
        if self.file_index != '0':
            plt.savefig(self.file_manager.fastFourierDir + "/" + "Spettro_frequenza_accelerazione_filtrata.png")
        plt.show()

        # Ritorno al dominio del tempo usando la IFFT (non più necessaria, dato che filtriamo nel tempo)
        self.ifft_signal = np.real(np.fft.ifft(fft_acceleration, axis=0))

        # Visualizzazione del segnale filtrato con FFT
        plt.figure(figsize=(10, 4))
        plt.plot(self.timestamp, self.filtered_Acc[:, 0], label="Segnale filtrato con Butterworth x", linewidth=2)
        plt.plot(self.timestamp, self.filtered_Acc[:, 1], label="Segnale filtrato con Butterworth y", linewidth=2)
        plt.plot(self.timestamp, self.filtered_Acc[:, 2], label="Segnale filtrato con Butterworth z", linewidth=2)
        plt.title("Segnale dopo il filtraggio Butterworth")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Accelerazione")
        plt.legend()
        plt.grid()
        if self.file_index != '0':
            plt.savefig(self.file_manager.fastFourierDir + "/" + "Segnale_Filtrato_Butterworth.png")
        plt.show()







