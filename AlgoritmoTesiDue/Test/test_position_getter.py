import pytest
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

# Assicura che Python trovi PositionGetter.py
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

TestPath="../AlgoritmoTesiDue/MediaTest/"

from ..PositionGetter import PositionGetter

# Creazione di dati di test finti
@pytest.fixture
def fake_data():
    directory = "SensorLogger"
    file_index = "Test_DataDue"
    sample_rate = 100

    # Costruzione del percorso corretto
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', directory, 'file_uniti', f"{file_index}.csv"))

    print(f"Sto cercando il file in: {csv_path}")  # Debug del percorso

    try:
        DataFrame = pd.read_csv(csv_path, delimiter=',', na_values=[''], encoding="utf-8")
    except FileNotFoundError:
        pytest.fail(f"Errore: il file CSV {csv_path} non esiste.")

    # Rimuove spazi bianchi e converte i dati a numerico
    DataFrame = DataFrame.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    DataFrame = DataFrame.apply(pd.to_numeric, errors='coerce')

    # Controlla se ci sono dati mancanti
    if DataFrame.isna().sum().sum() > 0:
        print("Attenzione: il dataset contiene valori NaN.")

    DataFrame = DataFrame.to_numpy()

    # Estrarre i dati
    timestamp = DataFrame[:, 0]
    accelerometerData = DataFrame[:, 1:4]
    gyroscopeData = DataFrame[:, 7:10]
    orientationData = DataFrame[:, 4:7]
    magnetometerData = DataFrame[:, 12:15]
    barometerData = DataFrame[:, 10:12]

    return timestamp, accelerometerData, gyroscopeData, orientationData, magnetometerData, sample_rate, file_index

# Test del costruttore (__init__)
def test_init(fake_data):
    timestamp, acc, gyro, orient, mag, sample_rate, file_index = fake_data
    pg = PositionGetter(timestamp, acc, gyro, orient, mag, sample_rate, file_index)

    assert len(pg.timestamp) == len(timestamp)
    assert pg.Acc.shape == acc.shape
    assert pg.Gyro.shape == gyro.shape
    assert pg.Orient.shape == orient.shape
    assert pg.Mag.shape == mag.shape
    assert pg.sample_rate == sample_rate
    assert pg.file_index == file_index

# 2.Test della funzione getELA: verifico dimensione del vettore earthAcc
def test_getELA_earthAcc_dimension(fake_data):
    timestamp, acc, gyro, orient, mag, sample_rate, file_index = fake_data
    pg = PositionGetter(timestamp, acc, gyro, orient, mag, sample_rate, file_index)
    
    pg.getELA()  # Esegui la funzione
    
    assert pg.earthAcc.shape == (len(timestamp), 3)  # Verifica la dimensione dell'output

# 3.Test della funzione getELA: verifico se si ottiene con Orientamento nullo lo stesso risultato del vettore Acc
def test_getELA_identity_rotation(fake_data):
    timestamp, acc, gyro, orient, mag, sample_rate, file_index = fake_data
    pg = PositionGetter(timestamp, acc, gyro, orient, mag, sample_rate, file_index)
    
    # Azzeriamo l'orientazione per simulare un sistema senza rotazione
    pg.Orient = np.zeros_like(pg.Orient)
    
    pg.getELA()
    
    np.testing.assert_almost_equal(pg.earthAcc, pg.Acc, decimal=5)  # Deve essere uguale a Acc senza rotazione
    
# 4.Test del metodo getELA: verifico i valori contenuti nel vettore delta_time
def test_getELA_delta_time(fake_data):
    timestamp, acc, gyro, orient, mag, sample_rate, file_index = fake_data
    pg = PositionGetter(timestamp, acc, gyro, orient, mag, sample_rate, file_index)
    
    pg.getELA()
    
    assert pg.delta_time[0] == 0  # Il primo valore deve essere 0 (non può esserci differenza con sé stesso)
    assert np.all(pg.delta_time >= 0)  # Tutti i delta_time devono essere >= 0

# 5.Test del metodo getEla: verifico la consistenza dei delta time intorno al 0.01 (adattabile alla frequenza di campionamento)    
def test_getELA_delta_time_consistency(fake_data):
    timestamp, acc, gyro, orient, mag, sample_rate, file_index = fake_data
    pg = PositionGetter(timestamp, acc, gyro, orient, mag, sample_rate, file_index)

    pg.getELA()
    
    expected_dt = 1 / sample_rate
    tolerance = 0.002  # Massimo scostamento accettabile

    assert np.all((pg.delta_time[1:] >= expected_dt - tolerance) & 
                  (pg.delta_time[1:] <= expected_dt + tolerance)), \
        f"Delta time fuori range: valori minimi {pg.delta_time.min()}, massimi {pg.delta_time.max()}"
   
    
# 6.Test della funzione identify_moving_periods: verifico che il vettore di movimento sia della stessa grandezza del vettore timestamp e che contenga solo
#   valori booleani    
def test_identify_moving_periods(fake_data):
    timestamp, acc, gyro, orient, mag, sample_rate, file_index = fake_data
    pg = PositionGetter(timestamp, acc, gyro, orient, mag, sample_rate, file_index)
    
    pg.getELA()  # Prima calcoliamo l'accelerazione terrestre
    pg.identify_moving_periods()  # Esegui il metodo
    
    assert pg.is_moving.shape == (len(timestamp),)  # Deve essere un vettore della stessa lunghezza di timestamp
    assert np.all((pg.is_moving == 0) | (pg.is_moving == 1))  # Deve contenere solo 0 o 1
    
# 7.Test del metodo getELA: verifico il suo comportamento con una rotazione semplice nota
def test_getELA_simple_rotation():
    timestamp = np.array([0, 1])  # Due istanti di tempo
    acc = np.array([[1, 0, 0], [1, 0, 0]])  # Accelerazione lungo X
    gyro = np.zeros((2, 3))  # Nessun movimento angolare
    orient = np.array([[0, np.pi/2, 0], [0, np.pi/2, 0]])  # Rotazione di 90° attorno all'asse Y
    mag = np.zeros((2, 3))  # Non rilevante per questo test
    sample_rate = 1
    file_index = 0
    
    pg = PositionGetter(timestamp, acc, gyro, orient, mag, sample_rate, file_index)
    pg.getELA()
    
    expected_result = np.array([[0, 0, -1], [0, 0, -1]])  # Il vettore (1,0,0) dovrebbe diventare (0,0,-1)
    np.testing.assert_almost_equal(pg.earthAcc, expected_result, decimal=5)
    
# 8.Test della funzione getPosition: verifico che velocità e posizione abbiano lo stesso numero di elementi del timestamp
def test_getPositionData_output_shape(fake_data):
    timestamp, acc, gyro, orient, mag, sample_rate, file_index = fake_data
    pg = PositionGetter(timestamp, acc, gyro, orient, mag, sample_rate, file_index)
    
    pg.getELA()
    pg.identify_moving_periods()
    
    pg.getPositionData(pg.earthAcc, "TestData")

    assert pg.velocity.shape == (len(timestamp), 3)
    assert pg.position.shape == (len(timestamp), 3)
 
# 9.Test del metodo getPosition: verifico che la velocità sia uguale a [0,0,0] all'inizio del moto
def test_getPositionData_initial_velocity(fake_data):
    timestamp, acc, gyro, orient, mag, sample_rate, file_index = fake_data
    pg = PositionGetter(timestamp, acc, gyro, orient, mag, sample_rate, file_index)

    pg.getELA()
    pg.identify_moving_periods()
    pg.getPositionData(pg.earthAcc, "TestData")

    np.testing.assert_array_equal(pg.velocity[0], np.zeros(3))

# 10.Test del metodo getPosition: verifico che in assenza di accelerazione la velocità rimanga costante
def test_getPositionData_constant_velocity():
    timestamp = np.linspace(0, 10, 100)  # Simuliamo 100 campioni in 10 secondi
    acc = np.zeros((100, 3))  # Nessuna accelerazione
    gyro = np.zeros((100, 3))
    orient = np.zeros((100, 3))
    mag = np.zeros((100, 3))
    sample_rate = 10  # 10 Hz
    file_index = 0

    pg = PositionGetter(timestamp, acc, gyro, orient, mag, sample_rate, file_index)
    pg.getELA()
    pg.identify_moving_periods()
    pg.getPositionData(pg.earthAcc, "TestData")

    np.testing.assert_array_equal(pg.velocity, np.zeros((100, 3)))  # Velocità deve rimanere zero
  
# 11.Test del metodo getPosition: verifico in caso di accelerazione costante che la posizione incrementi col quadrato del tempo (legge oraria)
def test_getPositionData_linear_motion():
    
    timestamp = np.linspace(0, 10, 100)  # 100 campioni in 10 secondi
    acc = np.ones((100, 3))  # Accelerazione costante (1 m/s²)
    gyro = np.zeros((100, 3))
    orient = np.zeros((100, 3))
    mag = np.zeros((100, 3))
    sample_rate = 10
    file_index = 0

    pg = PositionGetter(timestamp, acc, gyro, orient, mag, sample_rate, file_index)
    pg.getELA()
    pg.identify_moving_periods()
    pg.getPositionData(pg.earthAcc, "TestData")

    v = np.zeros((100, 3)) 
    for i in range(1, 100):  
        v[i] = v[i-1] + np.ones(3) * pg.delta_time[i] 
    expected_position = np.zeros((100, 3))
    for i in range(1, 100):
        expected_position[i] =v[i]*pg.delta_time[i]  + 0.5 * pg.delta_time[i] **2
    
    plt.plot(timestamp, pg.position[:, 0], label="Calcolata X")
    plt.plot(timestamp, expected_position[:, 0], '--', label="Attesa X")
    plt.xlabel("Tempo (s)")
    plt.ylabel("Posizione (m)")
    plt.legend()
    plt.title("Confronto tra posizione calcolata e attesa")
    plt.savefig(TestPath + "AccelerazioneTest.png")
    plt.show()
    
    np.testing.assert_almost_equal(pg.position, expected_position, decimal=2)
    
