import pytest
import numpy as np
import matplotlib.pyplot as plt
from PositionGetters.PositionalDatas2 import PositionalDatas2
from scipy.fft import fft, ifft, fftfreq

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

# 2.Test filtro: verifico che se il segnale in input è costante, allora deve rimanere tale
def test_butter_lowpass_filter_constant_signal(positional_data_instance):
    
    data = np.ones((100, 3))  # Segnale costante su 3 assi
    filtered_data = positional_data_instance.butter_lowpass_filter(data, cutoff=5, fs=100)
    
    assert np.allclose(filtered_data, data)
# 3.Test filtro: verifico che se il segnale ha del white noise in input, venga ridotto dopo l'applicazione del filtro
def test_butter_lowpass_filter_white_noise_smoothing(positional_data_instance):
    
    np.random.seed(42)  # Per riproducibilità
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
   
# 8.Test filtro: verifico il comportamento del filtro aglli estremi dell'ordine, per verificare che non vengano generati dei ringing sostenuti
def test_butter_lowpass_filter_stability_and_behavior_extreme_orders(positional_data_instance2):
    fs = 1000  # Frequenza di campionamento (Hz)
    cutoff = 50  # Frequenza di cutoff (Hz)
    t = np.linspace(0, 1, fs, endpoint=False)  # 1 secondo di dati
    signal = np.sin(2 * np.pi * 10 * t)  # Segnale sinusoidale a 10 Hz (sotto il cutoff, dovrebbe passare senza problemi)
    data = signal.reshape(-1, 1)  # Converti in formato compatibile

    # Applica il filtro con ordine 1
    filtered_data_order_1 = positional_data_instance2.butter_lowpass_filter(data, cutoff=cutoff, fs=fs, order=1)

    # Applica il filtro con ordine 10
    filtered_data_order_10 = positional_data_instance2.butter_lowpass_filter(data, cutoff=cutoff, fs=fs, order=10) # potrebbe generare ringing

    # --- Verifico ---
    # che l'ordine 1 non introduca oscillazioni
    # piuttosto che verificare localmente il segnale con la derivata, controllo la differenza tra segnale originale e quello filtrato (controllo globale)
    max_relative_difference = np.max(np.abs(filtered_data_order_1[:, 0] - data[:, 0])) 
    assert max_relative_difference < 0.05

    # che l'ordine 10 non renda il segnale instabile
    assert np.all(np.isfinite(filtered_data_order_10))

    # --- PLOTTING ---
    plt.figure(figsize=(10, 4))
    plt.plot(t, data[:, 0], label="Segnale Originale", alpha=0.5)
    plt.plot(t, filtered_data_order_1[:, 0], label="Filtro Order=1", linewidth=2)
    plt.plot(t, filtered_data_order_10[:, 0], label="Filtro Order=10", linewidth=2)
    plt.title("Comportamento del Filtro con Ordini Estremi")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Ampiezza")
    plt.legend()
    plt.grid()
    plt.savefig(TestPath + "test7filtro_Comportamento_del_Filtro_con_Ordini_Estremi.png")
    plt.show()
    
# 9.Test filtro: verifico con un segnale noto se il filtro aggiunge o meno distorsioni di fase
def test_butter_lowpass_filter_no_phase_distortion(positional_data_instance2):
    # Generazione di un segnale di test
    fs = 100  # Frequenza di campionamento (Hz)
    t = np.linspace(0, 10, fs * 10)  # 10 secondi di dati
    f1, f2 = 1, 10  # Frequenze principali del segnale
    
    # SEGNALI DI TEST DIVERSI
    # 0.Segnale di prova con due componenti sinusoidali
    original_signal = np.column_stack([
        np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t),
        np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t),
        np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    ])
    
    # 1. Segnale con componente ad alta frequenza
    f1, f2 = 1, 20  # Frequenze diverse, una sopra il cutoff
    high_freq_signal = np.column_stack([
        np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t),
        np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t),
        np.sin(2 * np.pi * f1 * t) + np.sin(2 * np.pi * f2 * t)
    ])

    # 2. Segnale impulsivo (onda quadra)
    impulse_signal = np.sign(np.sin(2 * np.pi * f1 * t))
    impulse_signal = np.column_stack([impulse_signal, impulse_signal, impulse_signal])

    # 3. Rumore bianco
    np.random.seed(42)
    noise_signal = np.random.normal(0, 1, (len(t), 3))

    # 4. Segnale modulato AM
    carrier_freq = 10  # Portante a 10 Hz
    modulating_freq = 1  # Modulante a 1 Hz
    am_signal = (1 + 0.5 * np.sin(2 * np.pi * modulating_freq * t)) * np.sin(2 * np.pi * carrier_freq * t)
    am_signal = np.column_stack([am_signal, am_signal, am_signal])
    
    # Applicazione del filtro
    cutoff = 5  # Frequenza di taglio in Hz
    
    # Test su diversi segnali
    for signal, title in zip([original_signal, high_freq_signal, impulse_signal, noise_signal, am_signal],
                         ["Segnale sinusoidale","High Frequency Signal", "Impulse Signal", "White Noise", "AM Signal"]):
        filtered_signal = positional_data_instance2.butter_lowpass_filter(signal, cutoff, fs)

        # FFT del segnale filtrato
        fft_signal = fft(filtered_signal, axis=0)

        # IFFT per tornare nel dominio del tempo
        ifft_signal = np.real(ifft(fft_signal, axis=0))

        # Verifica della distorsione di fase
        plt.figure(figsize=(10, 5))
        plt.plot(t, signal[:, 0], label='Original Signal', linestyle='dashed', alpha=0.7)
        plt.plot(t, ifft_signal[:, 0], label='Reconstructed Signal (IFFT)', linewidth=2)
        plt.title(f"Phase Distortion Check:{title}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid()
        plt.savefig(TestPath + "test8filtro_Distorsioni_di_fase.png")
        plt.show()

        # Calcolo della differenza di fase tra il segnale originale e quello ricostruito
        diff_phase = original_signal - ifft_signal
        phase_shift = np.mean(diff_phase, axis=0)  # Media delle differenze per ogni asse

        print("Phase Shift Analysis:", phase_shift)
        # Assert per verificare che la distorsione di fase sia contenuta entro un margine accettabile
        assert np.all(np.abs(phase_shift) < 0.1), f"Phase shift too large for {title}: {phase_shift}"






