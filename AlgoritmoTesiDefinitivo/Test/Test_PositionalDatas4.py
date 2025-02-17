import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PositionGetters.PositionalDatas4 import PositionalDatas4

@pytest.fixture
def neural_network():
    """
    Fixture per inizializzare la rete neurale e i percorsi dei dati di test.
    """
    return PositionalDatas4.NeuralNetwork(
        test_X_path="SensorLogger/Training/girotondo_biblio.csv",
        test_y_path="SensorLogger/Training/p_girotondo_biblio_recostructed.csv",
        media_path="SensorLogger/Training"
    )

def test_model_training_performance(neural_network):
    """
    Testa il modello sui dati di training per verificare se ha appreso correttamente.
    """

    # Carica il modello addestrato
    model = keras.models.load_model(neural_network.model_path)

    # Carica i dati di training
    X_train = neural_network.load_data(neural_network.X_paths)
    y_train = neural_network.load_data(neural_network.y_paths)

    # Normalizzazione
    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    
    X_train_scaled = scaler_X.fit_transform(X_train)
    y_train_scaled = scaler_y.fit_transform(y_train)

    # Predizioni
    predicted_y_scaled = model.predict(X_train_scaled)
    predicted_y = scaler_y.inverse_transform(predicted_y_scaled)

    # Calcolo metriche
    mae = mean_absolute_error(y_train, predicted_y)
    mse = mean_squared_error(y_train, predicted_y)
    rmse = np.sqrt(mse)

    # Plot dei risultati
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
        
    ax.scatter(y_train[:, 0], y_train[:, 1], y_train[:, 2], c='blue', label='Original Y')
    ax.scatter(predicted_y[:, 0], predicted_y[:, 1], predicted_y[:, 2], c='red', label='Predicted Y')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Confronto tra Y Originale e Y Predetto')
    plt.show()
    
    # Stampa delle metriche per debugging
    print(f"Test Performance: MAE={mae}, MSE={mse}, RMSE={rmse}")

    # Imposta delle soglie per determinare se il modello ha appreso correttamente
    assert mae < 0.05, f"MAE troppo alto: {mae}"
    assert mse < 0.01, f"MSE troppo alto: {mse}"
    assert rmse < 0.1, f"RMSE troppo alto: {rmse}"
