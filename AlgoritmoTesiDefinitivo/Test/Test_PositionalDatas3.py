import pytest
import numpy as np
from PositionGetters.PositionalDatas3 import PositionalDatas3, KalmanFilter  # Sostituisci con il modulo corretto

TestPath="../AlgoritmoTesiDefinitivo/MediaTest/"

def generate_mock_data(num_samples=100, sample_rate=50):
    """Genera dati fittizi per testare PositionalDatas3"""
    timestamp = np.linspace(0, num_samples / sample_rate, num_samples)
    accelerometer_data = np.random.rand(num_samples, 3)
    orientation_data = np.random.rand(num_samples, 3)
    magnetometer_data = np.random.rand(num_samples, 3)
    file_index = 1
    directory = "test_dir"
    test = "unit_test"
    return timestamp, accelerometer_data, orientation_data, sample_rate, file_index, directory, test, magnetometer_data

@pytest.fixture
def positional_data_instance():
    """Istanza della classe PositionalDatas3 per test"""
    data = generate_mock_data()
    return PositionalDatas3(*data)

# 1.Test kalman:verifico il comportamento del filtro con input senza rumore
def test_kalman_no_noise():
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.zeros((9,9))  # Nessun rumore di processo
    R = np.zeros((9,9))  # Nessun rumore di osservazione
    P = np.eye(9)
    x0 = np.ones(9)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    u = np.zeros(9)
    z = np.ones(9)
    
    kf.predict(u)
    kf.update(z)
    
    assert np.allclose(kf.get_state(), np.ones(9))

# 2.Test kalman: verifico il comportamento del filtro con misurazioni costanti
def test_kalman_constant_measurements():
    
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 0.01
    R = np.eye(9) * 0.1
    P = np.eye(9)
    x0 = np.zeros(9)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    z = np.full(9, 5.0)
    for _ in range(50):
        kf.predict(np.zeros(9))
        kf.update(z)
    assert np.allclose(kf.get_state(), z, atol=0.1)

# 3.Test kalman: verifico il comportamento del filtro con outlier nelle misurazioni
def test_kalman_outlier_rejection():
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 0.01
    R = np.eye(9) * 10.0  # Aumentiamo il rumore di osservazione per mitigare gli outlier
    P = np.eye(9)
    x0 = np.zeros(9)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    z = np.zeros(9)
    
    # Aggiorniamo con valori normali
    for _ in range(10):
        kf.predict(np.zeros(9))
        kf.update(z)
    
    # Introduciamo un outlier
    z_outlier = np.full(9, 1000.0)
    kf.update(z_outlier)
    
    # Stato finale deve essere ancora vicino a zero e non influenzato drasticamente dall'outlier
    assert np.all(kf.get_state() < 100)

# 4.Test kalman: verifico che lo stato iniziale venga mantenuto correttamente e che converga ai valori attesi
def test_kalman_initial_state():
    
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 0.01
    R = np.eye(9) * 0.1
    P = np.eye(9)
    x0 = np.full(9, 2.0)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    assert np.allclose(kf.get_state(), np.full(9, 2.0))

# 5.Test kalman: verifico l'effetto di covarianze iniziali grandi e piccole sulla stabilità del filtro
def test_kalman_covariance_variations():
    
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 0.01
    R = np.eye(9) * 0.1
    
    # Covarianza molto grande
    P_large = np.eye(9) * 1000
    kf_large = KalmanFilter(F, B, H, Q, R, P_large, np.zeros(9))
    
    # Covarianza molto piccola
    P_small = np.eye(9) * 0.0001
    kf_small = KalmanFilter(F, B, H, Q, R, P_small, np.zeros(9))
    
    # Controlliamo che la covarianza grande faccia convergere più lentamente
    for _ in range(50):
        kf_large.predict(np.zeros(9))
        kf_large.update(np.full(9, 1.0))
    
    for _ in range(50):
        kf_small.predict(np.zeros(9))
        kf_small.update(np.full(9, 1.0))
    
    assert np.allclose(kf_small.get_state(), np.full(9, 1.0), atol=0.1)
    assert np.allclose(kf_large.get_state(), np.full(9, 1.0), atol=1.0)
    
# 6.Test kalman: verifico il comportamento del filtro con alto rumore di processo    
def test_kalman_high_process_noise():
    
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 10.0  # Alto rumore di processo
    R = np.eye(9) * 0.1
    P = np.eye(9)
    x0 = np.zeros(9)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    z = np.zeros(9)
    
    for _ in range(50):
        kf.predict(np.zeros(9))
        kf.update(z)
    
    assert np.linalg.norm(kf.get_state()) < 5

# 7.Test kalman: verifico il comportamento del filtro con misure oscillanti
def test_kalman_varying_observations():
    
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 0.01
    R = np.eye(9) * 0.1
    P = np.eye(9)
    x0 = np.zeros(9)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    
    for i in range(20):
        z = np.sin(i * np.pi / 10) * np.ones(9)  # Misurazioni oscillanti
        kf.predict(np.zeros(9))
        kf.update(z)
    
    assert np.abs(kf.get_state()[0]) < 1.0
   
# 8.Test kalman: verifico il comportamento del filtro con sistema divergente per verificare la stabilità    
def test_kalman_diverging_system():
    
    F = np.eye(9) * 1.1  # Fattore di crescita che simula una dinamica instabile
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 0.01
    R = np.eye(9) * 0.1
    P = np.eye(9)
    x0 = np.zeros(9)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    
    for _ in range(50):
        kf.predict(np.zeros(9))
        kf.update(np.zeros(9))
    
    assert np.linalg.norm(kf.get_state()) < 100

# 9.Test kalman: verifico il comportamento del filtro con dati mancanti (NaN) per verificare la robustezza
def test_kalman_missing_data():
    
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 0.01
    R = np.eye(9) * 0.1
    P = np.eye(9)
    x0 = np.zeros(9)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    z = np.zeros(9)
    z[3] = np.nan  # Introduciamo un valore mancante
    
    kf.predict(np.zeros(9))
    
    try:
        kf.update(z)
        passed = True
    except Exception:
        passed = False
    
    assert passed

# 10.Test kalman: verifico il comportamento del filtro con misure quasi costanti per verificare la risposta del filtro
def test_kalman_low_information():
    
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 0.01
    R = np.eye(9) * 0.1
    P = np.eye(9)
    x0 = np.zeros(9)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    
    for _ in range(50):
        z = np.full(9, 0.01)  # Misurazioni quasi costanti
        kf.predict(np.zeros(9))
        kf.update(z)
    
    assert np.linalg.norm(kf.get_state() - z) < 0.1

# 11.Test kalman: verifico il comportamento del filtro con cambiamenti improvvisi nelle misure per verificare la reattività
def test_kalman_shock_input():
    
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 1.0  # Aumentiamo il rumore di processo per migliorare la reattività
    R = np.eye(9) * 0.5  # Aumentiamo il rumore di osservazione per bilanciare
    P = np.eye(9)
    x0 = np.zeros(9)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    
    for _ in range(10):
        kf.predict(np.zeros(9))
        kf.update(np.zeros(9))
    
    shock = np.full(9, 100.0)
    for _ in range(5):  # Iteriamo più volte per consentire la convergenza
        kf.predict(np.zeros(9))
        kf.update(shock)
    
    assert np.linalg.norm(kf.get_state() - shock) < 20

# 12.Test kalman: verifico il comportamento del filtro con elevato rumore per verificare la stabilità del filtro
def test_kalman_high_noise():
    
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 10.0  # Rumore di processo molto alto
    R = np.eye(9) * 10.0  # Rumore di osservazione molto alto
    P = np.eye(9)
    x0 = np.zeros(9)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    
    for _ in range(50):
        noisy_measurement = np.random.randn(9) * 10  # Simulazione di osservazioni molto rumorose
        kf.predict(np.zeros(9))
        kf.update(noisy_measurement)
    
    assert np.linalg.norm(kf.get_state()) < 100

# 13.Test kalman: verifico il comportamento del filtro con traiettoria complessa per verificare la capacità di tracking
def test_kalman_complex_trajectory():
   
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 0.01
    R = np.eye(9) * 0.1
    P = np.eye(9)
    x0 = np.zeros(9)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    
    for t in range(50):
        trajectory = np.array([np.sin(t/5), np.cos(t/5), np.sin(t/10)] * 3)  # Simula un movimento complesso
        kf.predict(np.zeros(9))
        kf.update(trajectory)
    
    assert np.linalg.norm(kf.get_state() - trajectory) < 1.0

# 14.Test kalman: verifico il tempo di esecuzione del filtro di Kalman su un dataset grande
def test_kalman_performance():
   
    import time
    F = np.eye(9)
    B = np.eye(9)
    H = np.eye(9)
    Q = np.eye(9) * 0.01
    R = np.eye(9) * 0.1
    P = np.eye(9)
    x0 = np.zeros(9)
    
    kf = KalmanFilter(F, B, H, Q, R, P, x0)
    
    start_time = time.time()
    for _ in range(1000):
        kf.predict(np.zeros(9))
        kf.update(np.random.rand(9))
    end_time = time.time()
    
    assert end_time - start_time < 1.0

# Testa se il filtro di Kalman è applicato correttamente 
import matplotlib.pyplot as plt

def test_applicateKalman(positional_data_instance):
    """Testa se il filtro di Kalman riduce il rumore in modo efficace senza smussare troppo il segnale, visualizzando i dati."""
    
    # Introduciamo rumore artificiale nei dati originali
    noise = np.random.normal(0, 1, positional_data_instance.Acc.shape) * 3  # Rumore moderato
    positional_data_instance.Acc += noise  # Aggiungiamo il rumore ai dati
    
    # Applichiamo il filtro di Kalman
    positional_data_instance.applicateKalman()
    
    # Controllo che le matrici siano state inizializzate correttamente
    assert positional_data_instance.kalman_acc.shape == positional_data_instance.Acc.shape
    assert positional_data_instance.kalman_orient.shape == positional_data_instance.Orient.shape
    assert positional_data_instance.kalman_mag.shape == positional_data_instance.Mag.shape
    
    # Estriamo i dati per il grafico
    time = positional_data_instance.timestamp
    acc_original = positional_data_instance.Acc[:, 0]  # Accelerazione originale su asse X
    acc_filtered = positional_data_instance.kalman_acc[:, 0]  # Accelerazione filtrata su asse X

    # Calcoliamo la media mobile come riferimento
    window_size = 5
    rolling_mean = np.convolve(acc_original, np.ones(window_size)/window_size, mode='valid')

    # Creiamo il grafico per visualizzare i dati originali e filtrati
    plt.figure(figsize=(12, 6))
    plt.plot(time, acc_original, label="Accelerazione Originale", alpha=0.5, color='gray')
    plt.plot(time[:len(rolling_mean)], rolling_mean, label="Media Mobile", linestyle='dashed', color='orange')
    plt.plot(time, acc_filtered, label="Accelerazione Filtrata (Kalman)", color='blue')

    plt.title("Confronto tra Dati Originali, Media Mobile e Filtro di Kalman")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Accelerazione (m/s^2)")
    plt.legend()
    plt.grid()
    
    # Mostriamo il grafico
    plt.savefig(TestPath + "test15kalman_Confronto_Dati_Originali_MediaMobile_Filtro_di_Kalman.png")
    plt.show()

    # Calcoliamo la varianza dei dati originali e filtrati
    std_original = np.std(positional_data_instance.Acc, axis=0)
    std_filtered = np.std(positional_data_instance.kalman_acc, axis=0)

    # Verifica della riduzione del rumore
    assert np.all(std_filtered < std_original), \
        f"Il filtro di Kalman non ha ridotto la varianza! (Originale: {std_original}, Filtrato: {std_filtered})"






