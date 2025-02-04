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
        self.getPositionData(self.Acc,"PositionalData2")
        self.plotGraphics("Accelerazione_after_IFFT","Angoli_di_Eulero",self.ifft_signal,self.Orient)

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
        plt.plot(self.timestamp, self.Acc[:, 0], label="x")
        plt.plot(self.timestamp, self.Acc[:, 1], label="y")
        plt.plot(self.timestamp, self.Acc[:, 2], label="z")
        plt.title("Accelerazione lineare terrestre (dominio del tempo)")
        plt.xlabel("Tempo (s)")
        plt.ylabel("Accelerazione lineare terrestre")
        plt.legend()
        if self.file_index != '0':                                      # '0' è il file_index dei test
            plt.savefig(self.file_manager.fastFourierDir + "/" + "Accelerazione_lineare_terrestre.png")
        plt.show()

        # Applicazione della FFT per ciascun asse
        fft_acceleration = np.fft.fft(self.Acc, axis=0)
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
        plt.plot(self.timestamp, self.Acc[:, 0], label="Segnale originale x", alpha=0.5)
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




