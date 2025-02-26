import pytest
import numpy as np
import pandas as pd
from unittest.mock import MagicMock, patch
from PositionGetters.PositionalDatas5 import PositionalDatas5

@pytest.fixture
def sample_data():
    """ Crea dati di test simulati per inizializzare la classe PositionalDatas5 """
    timestamp = np.array([1, 2, 3])
    acc = np.random.rand(3, 3)
    gyro = np.random.rand(3, 3)
    orient = np.random.rand(3, 3)
    mag = np.random.rand(3, 3)
    press = np.random.rand(3, 1)
    ela = np.random.rand(3, 1)
    ifft = np.random.rand(3, 1)
    pd1 = np.random.rand(3, 1)
    pd3 = np.random.rand(3, 1)
    return timestamp, acc, orient, 100, 1, "dir", True, mag, gyro, press, ela, ifft, pd1, pd3

def test_initialization(sample_data):
    """ Testa l'inizializzazione della classe PositionalDatas5 """
    pos_data = PositionalDatas5(*sample_data)

    assert pos_data.Mag.shape == (3, 3)
    assert pos_data.Gyro.shape == (3, 3)
    assert pos_data.Press.shape == (3, 1)
    assert pos_data.ELA.shape == (3, 1)
    assert pos_data.ifft.shape == (3, 1)
    assert pos_data.PD1.shape == (3, 1)
    assert pos_data.PD3.shape == (3, 1)

@patch("PositionGetters.PositionalDatas5.PositionalDatas5.NeuralNetwork")
def test_processData(mock_neural_network, sample_data):
    """ Testa che il metodo processData richiami i metodi corretti """
    pos_data = PositionalDatas5(*sample_data)
    pos_data.file_manager = MagicMock()
    
    mock_nn_instance = mock_neural_network.return_value
    mock_nn_instance.train_model = MagicMock()
    mock_nn_instance.predict_new_data = MagicMock()
    mock_nn_instance.predicted_y = np.random.rand(3, 3)

    pos_data.processData()

    # Controlla se il metodo della rete neurale è stato chiamato
    mock_nn_instance.train_model.assert_called_once()
    mock_nn_instance.predict_new_data.assert_called_once()
    pos_data.file_manager.save_position_data.assert_called_once()

@pytest.fixture
def neural_network_instance(sample_data):
    """ Crea un'istanza della classe interna NeuralNetwork con dati simulati """
    test_X = np.random.rand(3, 27)  # Simuliamo un dataset di input con 27 feature (invece di 10)
    media_path = "MediaTest"
    file_index = 1
    nn = PositionalDatas5.NeuralNetwork(test_X, media_path, file_index)
    return nn

def test_load_data(neural_network_instance):
    """ Testa che il metodo load_data carichi correttamente i dati """
    with patch("pandas.read_csv") as mock_read_csv:
        mock_read_csv.return_value = pd.DataFrame(np.random.rand(3, 3))

        loaded_data = neural_network_instance.load_data(["fake_path.csv"])
        assert loaded_data.shape == (3, 3)

def test_preprocess_data(neural_network_instance):
    """ Testa che il metodo preprocess_data normalizzi correttamente i dati """
    X = np.random.rand(3, 5)
    y = np.random.rand(3, 3)

    X_scaled, y_scaled = neural_network_instance.preprocess_data(X, y)

    assert X_scaled.shape == X.shape
    assert y_scaled.shape == y.shape

@patch("tensorflow.keras.Model.fit")
@patch("tensorflow.keras.Model.save")
def test_train_model(mock_save, mock_fit, neural_network_instance):
    """ Testa che il metodo train_model richiami il training della rete neurale """
    mock_fit.return_value = None  # Simula il training

    # Dati di input e output
    neural_network_instance.X_paths = np.random.rand(3, 10)
    with patch.object(neural_network_instance, "load_data", return_value=np.random.rand(3, 3)):
        neural_network_instance.train_model()

    mock_fit.assert_called_once()
    mock_save.assert_called_once_with(neural_network_instance.model_path)



