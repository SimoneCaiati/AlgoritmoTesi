import pytest
import numpy as np
import matplotlib.pyplot as plt
from PositionGetters.PositionalDatas2 import PositionalDatas2

TestPath="../AlgoritmoTesiDefinitivo/MediaTest/"

@pytest.fixture
def positional_data_instance():
    timestamp = np.linspace(0, 1, 100)  # Simuliamo 100 campioni temporali
    accelerometerData = np.random.randn(100, 3)  # Simuliamo dati di accelerazione casuali
    orientationData = np.random.randn(100, 3)  # Simuliamo dati di orientamento casuali
    sample_rate = 100  # Frequenza di campionamento 100 Hz
    file_index = 'TestFakeDataPD2'
    directory = "MediaTest"
    test = True
    
    return PositionalDatas2(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, test)

# 1.Test filtro: verifico che la dimensione di input coincida con quella di output
def test_butter_lowpass_filter_output_shape(positional_data_instance):
    data = np.random.randn(100, 3)  # Simuliamo dati casuali (100 campioni, 3 assi)
    filtered_data = positional_data_instance.butter_lowpass_filter(data, cutoff=5, fs=100)
    
    assert filtered_data.shape == data.shape

# 2.Test filtro: verifico che se il segnale in input � costante, allora deve rimanere tale
def test_butter_lowpass_filter_constant_signal(positional_data_instance):
    
    data = np.ones((100, 3))  # Segnale costante su 3 assi
    filtered_data = positional_data_instance.butter_lowpass_filter(data, cutoff=5, fs=100)
    
    assert np.allclose(filtered_data, data)
# 3.Test filtro: verifico che se il segnale ha del white noise in input, venga ridotto dopo l'applicazione del filtro
def test_butter_lowpass_filter_white_noise_smoothing(positional_data_instance):
    
    np.random.seed(42)  # Per riproducibilit�
    data = np.random.randn(100, 3)  # Rumore bianco
    filtered_data = positional_data_instance.butter_lowpass_filter(data, cutoff=5, fs=100)

    # Controlliamo che il filtro abbia ridotto la varianza (indicatore di smussamento)
    assert np.var(filtered_data) < np.var(data)

@pytest.fixture
def positional_data_instance2():
    timestamp = np.linspace(0, 1, 1000)  # Simuliamo 1000 campioni temporali (1 secondo a 1000 Hz)
    accelerometerData = np.random.randn(1000, 3)  
    orientationData = np.random.randn(1000, 3)  
    sample_rate = 1000  # Frequenza di campionamento 1000 Hz
    file_index = 'TestFakeDataPD1'
    directory = "MediaTest"
    test = True
    
    return PositionalDatas2(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, test)

# 4.Test filtro: verifico che il filtro mantenga le frequenze basse e attenui le alte
def test_butter_lowpass_filter_cutoff_frequency(positional_data_instance2):
    fs = 1000  # Frequenza di campionamento (Hz)
    cutoff = 50  # Frequenza di cutoff (Hz)
    
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 secondo di dati
    
    # Generiamo un segnale con due sinusoidi: 
    # - Una a 20 Hz (sotto il cutoff, dovrebbe passare)
    # - Una a 200 Hz (sopra il cutoff, dovrebbe essere attenuata)
    low_freq_signal = np.sin(2 * np.pi * 20 * t)  # 20 Hz
    high_freq_signal = np.sin(2 * np.pi * 200 * t)  # 200 Hz
    data = (low_freq_signal + high_freq_signal).reshape(-1, 1)  # Segnale combinato

    # Applichiamo il filtro passa-basso
    filtered_data = positional_data_instance2.butter_lowpass_filter(data, cutoff=cutoff, fs=fs, order=4)
    
    # Calcoliamo lo spettro FFT del segnale originale e filtrato
    fft_original = np.abs(np.fft.rfft(data[:, 0]))
    fft_filtered = np.abs(np.fft.rfft(filtered_data[:, 0]))
    freqs = np.fft.rfftfreq(len(t), d=1/fs)  # Asse delle frequenze

    # Troviamo gli indici delle frequenze specifiche (20 Hz e 200 Hz)
    idx_20Hz = np.argmin(np.abs(freqs - 20))
    idx_200Hz = np.argmin(np.abs(freqs - 200))

    # Verifica: la componente a 20 Hz deve rimanere invariata (entro un margine di errore)
    assert np.isclose(fft_filtered[idx_20Hz], fft_original[idx_20Hz], atol=0.1)

    # Verifica: la componente a 200 Hz deve essere attenuata in modo significativo
    assert fft_filtered[idx_200Hz] < 0.2 * fft_original[idx_200Hz]

    # --- PLOTTING ---

    # 1. Segnale originale e filtrato nel dominio del tempo
    plt.figure(figsize=(10, 4))
    plt.plot(t, data[:, 0], label="Segnale Originale", alpha=0.5)
    plt.plot(t, filtered_data[:, 0], label="Segnale Filtrato", linewidth=2)
    plt.title("Segnale Originale vs Filtrato (Dominio del Tempo)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza")
    plt.legend()
    plt.grid()
    plt.savefig(TestPath + "Test4filtro_Segnale_Originale_vs_Filtrato_Dominio_del_Tempo.png")
    plt.show()

    # 2. Spettro di frequenza prima e dopo il filtraggio
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_original, label="FFT Originale", alpha=0.5)
    plt.plot(freqs, fft_filtered, label="FFT Filtrata", linewidth=2)

    # Aggiunta della linea verticale per il cutoff
    plt.axvline(cutoff, color='red', linestyle='--', label="Frequenza di taglio (50 Hz)")

    plt.title("Spettro di Frequenza - Prima e Dopo il Filtraggio")
    plt.xlabel("Frequenza (Hz)")
    plt.ylabel("Ampiezza")
    plt.legend()
    plt.grid()
    plt.savefig(TestPath + "test4filtro_Spettro_di_Frequenza_Prima_e_Dopo_Filtraggio.png")
    plt.show()

# 5.Test filtro: genero un singolo tono sinusoidale sopra il cutoff e verifico che venga attenuato drasticamente
def test_butter_lowpass_filter_single_tone_high_frequency(positional_data_instance2):
    fs = 1000  # Frequenza di campionamento (Hz)
    cutoff = 50  # Frequenza di cutoff (Hz)
    frequency = 200  # Frequenza del tono sinusoidale (superiore al cutoff)
    
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 secondo di dati
    signal = np.sin(2 * np.pi * frequency * t)  # Onda sinusoidale a 200 Hz
    data = signal.reshape(-1, 1)  # Converti in formato compatibile
    
    # Applica il filtro passa-basso
    filtered_data = positional_data_instance2.butter_lowpass_filter(data, cutoff=cutoff, fs=fs, order=4)

    # Analisi FFT per verificare attenuazione
    fft_original = np.abs(np.fft.rfft(data[:, 0]))
    fft_filtered = np.abs(np.fft.rfft(filtered_data[:, 0]))
    freqs = np.fft.rfftfreq(len(t), d=1/fs)

    # Troviamo l'indice della frequenza di 200 Hz
    idx_200Hz = np.argmin(np.abs(freqs - frequency))

    # La componente a 200 Hz deve essere attenuata significativamente
    assert fft_filtered[idx_200Hz] < 0.2 * fft_original[idx_200Hz]

    # --- PLOTTING ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, data[:, 0], label="Segnale Originale (200 Hz)", alpha=0.5)
    plt.plot(t, filtered_data[:, 0], label="Segnale Filtrato", linewidth=2)
    plt.title("Segnale Sinusoidale Originale vs Filtrato (200 Hz)")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza")
    plt.legend()
    plt.grid()
    plt.savefig(TestPath + "test5filtro_Segnale_Sinusoidale_Originale_vs_Filtrato_200Hz.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(freqs, fft_original, label="FFT Originale", alpha=0.5)
    plt.plot(freqs, fft_filtered, label="FFT Filtrata", linewidth=2)
    plt.axvline(cutoff, color='red', linestyle='--', label="Cutoff 50 Hz")
    plt.title("Spettro di Frequenza - Prima e Dopo il Filtraggio")
    plt.xlabel("Frequenza (Hz)")
    plt.ylabel("Ampiezza")
    plt.legend()
    plt.grid()
    plt.savefig(TestPath + "test5filtro_Spettro_di_Frequenza_Prima_e_Dopo_Filtraggio.png")
    plt.show()

# 6.Test filtro: genero un segnale "rampa" la cui derivata dovrebbe essere una costante, e verifico che il filtro non produca alcuna 
#   oscillazione artificiale considerevole    
def test_butter_lowpass_filter_ramp_signal_response(positional_data_instance2):
    fs = 1000  # Frequenza di campionamento (Hz)
    cutoff = 50  # Frequenza di cutoff (Hz)
    
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 secondo di dati
    ramp_signal = np.linspace(-1, 1, fs)  # Segnale rampa crescente
    data = ramp_signal.reshape(-1, 1)  # Converti in formato compatibile
    
    # Applica il filtro passa-basso
    filtered_data = positional_data_instance2.butter_lowpass_filter(data, cutoff=cutoff, fs=fs, order=4)

    # Il filtro non deve introdurre oscillazioni artificiali
    diff_original = np.diff(data[:, 0])             # Derivata numerica del segnale originale
    diff_filtered = np.diff(filtered_data[:, 0])    # Derivata numerica del segnale filtrato
    
    # Controlliamo che la derivata non introduca variazioni significative
    assert np.max(np.abs(diff_filtered - diff_original)) < 0.01  # Soglia di variazione accettabile

    # --- PLOTTING ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, data[:, 0], label="Segnale Rampa Originale", alpha=0.5)
    plt.plot(t, filtered_data[:, 0], label="Segnale Filtrato", linewidth=2)
    plt.title("Segnale Rampa Originale vs Filtrato")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza")
    plt.legend()
    plt.grid()
    plt.savefig(TestPath + "test6filtro_Segnale_Rampa_Originale_vs_Filtrato.png")
    plt.show()

    plt.figure(figsize=(10, 4))
    plt.plot(t[:-1], diff_original, label="Derivata Originale", alpha=0.5)
    plt.plot(t[:-1], diff_filtered, label="Derivata Filtrata", linewidth=2)
    plt.title("Derivata Numerica: Prima e Dopo il Filtraggio")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Derivata")
    plt.legend()
    plt.grid()
    plt.savefig(TestPath + "test6filtro_Derivata_Numerica_Prima_e_Dopo_Filtraggio.png")
    plt.show()



