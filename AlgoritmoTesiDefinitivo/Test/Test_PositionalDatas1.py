import pytest
import numpy as np
import matplotlib.pyplot as plt
from PositionGetters.PositionalDatas1 import PositionalDatas1

TestPath="../AlgoritmoTesiDefinitivo/MediaTest/"

@pytest.fixture
def sample_data():
    """Fixture per creare dati di input di esempio"""
    timestamp = np.array([0.0, 0.1, 0.2, 0.3])  # 4 campioni di tempo
    accelerometerData = np.array([
        [0, 0, 9.81],
        [0, 0, 9.81],
        [0, 0, 9.81],
        [0, 0, 9.81]
    ])  # Accelerazione in m/s^2 (supponiamo in quiete)
    
    orientationData = np.array([
        [0, 0, 0],  # Nessuna rotazione
        [np.pi/2, 0, 0],  # 90° attorno a X
        [0, np.pi/2, 0],  # 90° attorno a Y
        [0, 0, np.pi/2]   # 90° attorno a Z
    ])  # Orientazione in radianti

    sample_rate = 10  # 10 Hz
    file_index = 'TestFakeDataPD1'
    directory = "MediaTest"

    return timestamp, accelerometerData, orientationData, sample_rate, file_index, directory
# 1.Test __init__
def test_init(sample_data):
    timestamp, accelerometerData, orientationData, sample_rate, file_index, directory = sample_data
    pd1 = PositionalDatas1(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory,True)

    assert np.array_equal(pd1.timestamp, timestamp)
    assert np.array_equal(pd1.Acc, accelerometerData)
    assert np.array_equal(pd1.Orient, orientationData)
    assert pd1.sample_rate == sample_rate
    assert pd1.file_index == file_index
    assert pd1.directory == directory

# 2.Test della funzione getELA: verifico dimensione del vettore earthAcc
def test_getELA_earthAcc_dimension(sample_data):
    timestamp, accelerometerData, orientationData, sample_rate, file_index, directory = sample_data
    pd1 = PositionalDatas1(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory,True)
    
    pd1.getELA()  # Esegui la funzione
    
    assert pd1.earthAcc.shape == (len(timestamp), 3)  # Verifica la dimensione dell'output

# 3.Test della funzione getELA: verifico se si ottiene con Orientamento nullo lo stesso risultato del vettore Acc
def test_getELA_identity_rotation(sample_data):
    timestamp, accelerometerData, orientationData, sample_rate, file_index, directory = sample_data
    pd1 = PositionalDatas1(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory,True)
    
    # Azzeriamo l'orientazione per simulare un sistema senza rotazione
    pd1.Orient = np.zeros_like(pd1.Orient)
    
    pd1.getELA()
    
    np.testing.assert_almost_equal(pd1.earthAcc, pd1.Acc, decimal=5)  # Deve essere uguale a Acc senza rotazione
    
# 4.Test del metodo getELA: verifico i valori contenuti nel vettore delta_time
def test_getELA_delta_time(sample_data):
    timestamp, accelerometerData, orientationData, sample_rate, file_index, directory = sample_data
    pd1 = PositionalDatas1(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory,True)
    
    pd1.getELA()
    
    assert pd1.delta_time[0] == 0  # Il primo valore deve essere 0 (non può esserci differenza con sé stesso)
    assert np.all(pd1.delta_time >= 0)  # Tutti i delta_time devono essere >= 0

# 5.Test del metodo getEla: verifico la consistenza dei delta time intorno al 0.01 (adattabile alla frequenza di campionamento)    
def test_getELA_delta_time_consistency(sample_data):
    timestamp, accelerometerData, orientationData, sample_rate, file_index, directory = sample_data
    pd1 = PositionalDatas1(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory,True)

    pd1.getELA()
    
    expected_dt = 1 / sample_rate
    tolerance = 0.002  # Massimo scostamento accettabile

    assert np.all((pd1.delta_time[1:] >= expected_dt - tolerance) & 
                  (pd1.delta_time[1:] <= expected_dt + tolerance)), \
        f"Delta time fuori range: valori minimi {pd1.delta_time.min()}, massimi {pd1.delta_time.max()}"

# 6.Test del metodo getPosition: verifico in caso di accelerazione costante che la posizione incrementi col quadrato del tempo (legge oraria)
def test_getPositionData_linear_motion():
    
    timestamp = np.linspace(0, 10, 100)  # 100 campioni in 10 secondi
    acc = np.ones((100, 3))  # Accelerazione costante (1 m/s²)
    orient = np.zeros((100, 3))
    sample_rate = 10
    file_index = '0'
    directory= '0'

    pg = PositionalDatas1(timestamp, acc, orient, sample_rate, file_index, directory,True)
    pg.getELA()
    pg.identify_moving_periods(pg.earthAcc)
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
    
# 7.Test getEla: verifico la rotazione sugli assi rotazionali di 90°
def test_getELA(sample_data):
    timestamp, accelerometerData, orientationData, sample_rate, file_index, directory = sample_data
    pd1 = PositionalDatas1(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory,True)

    pd1.getELA()

    # Controllo su angoli di rotazione semplici
    expected_acc_1 = np.array([0, 9.81, 0])  # Dopo rotazione 90° attorno a X
    expected_acc_2 = np.array([9.81, 0, 0])  # Dopo rotazione 90° attorno a Y
    expected_acc_3 = np.array([0, 0, 9.81])  # Dopo rotazione 90° attorno a Z (nessun cambiamento in Z)

    np.testing.assert_array_almost_equal(pd1.earthAcc[1], expected_acc_1, decimal=2)
    np.testing.assert_array_almost_equal(pd1.earthAcc[2], expected_acc_2, decimal=2)
    np.testing.assert_array_almost_equal(pd1.earthAcc[3], expected_acc_3, decimal=2)
    
# 8.Test getEla: verifico la rotazione sugli assi rotazionali di 180°
def test_getELA_180_degree_rotations(sample_data):
    timestamp, accelerometerData, orientationData, sample_rate, file_index, directory = sample_data
    pd1 = PositionalDatas1(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, True)

    # Modifica dell'orientazione per rotazioni di 180° sui 3 assi
    pd1.Orient = np.array([
        [0, 0, 0],  # Nessuna rotazione
        [np.pi, 0, 0],  # 180° attorno a X
        [0, np.pi, 0],  # 180° attorno a Y
        [0, 0, np.pi]   # 180° attorno a Z
    ])

    pd1.getELA()

    expected_acc_x = np.array([0, 0, -9.81])  # Dopo 180° attorno a X: Z diventa -Z
    expected_acc_y = np.array([0, 0, -9.81])  # Dopo 180° attorno a Y: X e Z si invertono
    expected_acc_z = np.array([0, 0, 9.81])  # Dopo 180° attorno a Z: nessun cambiamento lungo Z

    np.testing.assert_array_almost_equal(pd1.earthAcc[1], expected_acc_x, decimal=10)
    np.testing.assert_array_almost_equal(pd1.earthAcc[2], expected_acc_y, decimal=10)
    np.testing.assert_array_almost_equal(pd1.earthAcc[3], expected_acc_z, decimal=10)

# 9.Test getEla: verifico la rotazione sugli assi rotazionali di 45°
def test_getELA_45_degree_rotations(sample_data):
    timestamp, accelerometerData, orientationData, sample_rate, file_index, directory = sample_data
    pd1 = PositionalDatas1(timestamp, accelerometerData, orientationData, sample_rate, file_index, directory, True)

    # Rotazioni di 45° attorno agli assi X, Y e Z
    pd1.Orient = np.array([
        [0, 0, 0],  # Nessuna rotazione
        [np.pi / 4, 0, 0],  # 45° attorno a X
        [0, np.pi / 4, 0],  # 45° attorno a Y
        [0, 0, np.pi / 4]   # 45° attorno a Z
    ])

    pd1.getELA()

    sqrt2_div2 = np.sqrt(2) / 2  # sin(45°) e cos(45°) hanno lo stesso valore

    expected_acc_x = np.array([0, 9.81 * sqrt2_div2, 9.81 * sqrt2_div2])
    expected_acc_y = np.array([9.81 * sqrt2_div2, 0, 9.81 * sqrt2_div2])
    expected_acc_z = np.array([0 , 0, 9.81])

    np.testing.assert_array_almost_equal(pd1.earthAcc[1], expected_acc_x, decimal=2)
    np.testing.assert_array_almost_equal(pd1.earthAcc[2], expected_acc_y, decimal=2)
    np.testing.assert_array_almost_equal(pd1.earthAcc[3], expected_acc_z, decimal=2)
