import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from PositionGetters.PositionalDatas4 import PositionalDatas4

@pytest.fixture
def neural_network():
    """
    Fixture per inizializzare la rete neurale e i percorsi dei dati di test.
    """
    return PositionalDatas4.NeuralNetwork(
        test_X_path="SensorLogger/File_uniti/ADL4.csv",
        media_path="SensorLogger/MediaTest",
        file_index="ADL4",
        directory="SensorLogger"
    )

def test_model_training_performance(neural_network):
    """
    Testa il modello sui dati di test specifici per verificare se il modello ha appreso correttamente.
    """

    # Controlla se il modello esiste
    try:
        model = keras.models.load_model(neural_network.model_path, custom_objects={'mse': keras.losses.MeanSquaredError})
    except OSError:
        pytest.fail(f"Modello non trovato: {neural_network.model_path}. Assicurati di averlo addestrato!")

    # Carica i dati di test da file CSV
    X_test = pd.read_csv(neural_network.test_X_path, delimiter=',', na_values=['']).replace(" ", "").dropna().to_numpy().astype(float)
    y_test = pd.read_csv("SensorLogger/Training/p_ADL4_reconstructed.csv", delimiter=',', na_values=['']).replace(" ", "").dropna().to_numpy().astype(float)

    # Normalizzazione con i parametri di training
    scaler_X = RobustScaler()
    scaler_y = RobustScaler()

    X_test_scaled = scaler_X.fit_transform(X_test)
    y_test_scaled = scaler_y.fit_transform(y_test)

    # Predizioni
    predicted_y_scaled = model.predict(X_test_scaled)
    predicted_y = scaler_y.inverse_transform(predicted_y_scaled)
    predicted_y[predicted_y[:, 2] < 0.001, 2] = 0

    # Calcolo metriche
    mae = mean_absolute_error(y_test, predicted_y)
    mse = mean_squared_error(y_test, predicted_y)
    rmse = np.sqrt(mse)

    # Plot dei risultati
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
        
    ax.scatter(y_test[:, 0], y_test[:, 1], y_test[:, 2], c='blue', label='Original Y (Test)')
    ax.scatter(predicted_y[:, 0], predicted_y[:, 1], predicted_y[:, 2], c='red', label='Predicted Y')
        
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Confronto tra Y Originale e Y Predetto (Test Set)')
    plt.show()
    
     # Analizza la distribuzione di X, Y, Z
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1)
    plt.hist(y_test[:, 0], bins=50, alpha=0.7, label='X')
    plt.legend()
    plt.subplot(1, 3, 2)
    plt.hist(y_test[:, 1], bins=50, alpha=0.7, label='Y')
    plt.legend()
    plt.subplot(1, 3, 3)
    plt.hist(y_test[:, 2], bins=50, alpha=0.7, label='Z')
    plt.legend()
    plt.show()
    
    # Stampa delle metriche per debugging
    print(f"Test Performance: MAE={mae}, MSE={mse}, RMSE={rmse}")

    # Imposta delle soglie per determinare se il modello ha appreso correttamente
    assert mae < 0.1, f"MAE troppo alto: {mae}"
    assert mse < 0.02, f"MSE troppo alto: {mse}"
    assert rmse < 0.2, f"RMSE troppo alto: {rmse}"
