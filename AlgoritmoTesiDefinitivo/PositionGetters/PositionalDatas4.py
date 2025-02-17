import numpy as np
from PositionGetters.PositionalData import PositionalData 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

class PositionalDatas4(PositionalData):
    def __init__(self,timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test, magnetometerData, gyroscope, pressure, ELA):
        super().__init__(timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test,specificPD="PD4") 
        self.Mag = magnetometerData
        self.Gyro = gyroscope
        self.Press = pressure
        self.ELA = ELA
        

    def processData(self):
        self.timestamp = self.timestamp.reshape(-1, 1)
        dati = np.concatenate((self.timestamp, self.Acc, self.Mag, self.Gyro, self.Orient, self.Press), axis=1)
        nn = self.NeuralNetwork(dati, f"SensorLogger/PositionalDatas/{self.file_index}/Real.csv", self.file_manager.mediaDir)
        #nn.train_model()
        nn.predict_new_data()
        PositionDataFrame=pd.DataFrame(nn.predicted_y)
        self.file_manager.save_position_data(PositionDataFrame, "PositionalData4")
       

    class NeuralNetwork:
        def __init__(self,test_X_path,test_y_path, media_path):
            self.X_paths = ["SensorLogger/Training/girotondo_biblio.csv","SensorLogger/Training/9metri_dritto_destra.csv"]  # Lista dei path per X
            self.y_paths = ["SensorLogger/Training/p_girotondo_biblio_recostructed.csv","SensorLogger/Training/p_9metri_dritto_destra_reconstructed.csv"]  # Lista dei path per Y
            self.model_path = "SensorLogger/Training/trained_model.h5"  # Percorso per salvare il modello
            self.test_X_path = test_X_path  # Percorso del dataset di test X
            self.test_y_path = test_y_path  # Percorso del dataset di test Y
            self.media_path = media_path
    
        def load_data(self, paths):
            all_data = []
            for path in paths:
                data = pd.read_csv(path, delimiter=',', na_values=['']).replace(" ", "").dropna().to_numpy().astype(float)
                all_data.append(data)
            return np.vstack(all_data)
    
        def preprocess_data(self, X, y):
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_scaled = scaler_X.fit_transform(X)
            y_scaled = scaler_y.fit_transform(y)
        
            return X_scaled, y_scaled
    
        def train_model(self):
            # Carica dati da più dataset
            X = self.load_data(self.X_paths)
            y = self.load_data(self.y_paths)
        
            # Preprocessing
            X_scaled, y_scaled = self.preprocess_data(X, y)
        
            # Definizione della rete neurale
            input_layer = keras.layers.Input(shape=(X_scaled.shape[1],))
            common_layer = keras.layers.Dense(64, activation='relu')(input_layer)
            common_layer = keras.layers.Dense(32, activation='relu')(common_layer)
        
            output_layer = keras.layers.Dense(3, activation='linear')(common_layer)
        
            model = keras.Model(inputs=input_layer, outputs=output_layer)
        
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
            early_stopping = keras.callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)
            reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6)
        
            # Training su tutti i dati disponibili
            model.fit(X_scaled, y_scaled, epochs=500, batch_size=32, callbacks=[early_stopping, reduce_lr])
        
            # Salvataggio del modello
            model.save(self.model_path)
            print(f"Modello salvato in {self.model_path}")
    
        def predict_new_data(self):
            # Carica il modello addestrato
            model = keras.models.load_model(self.model_path)
        
            # Carica il dataset di test
            X_test = self.test_X_path
            y_test = self.load_data([self.test_y_path])
        
            # Normalizzazione
            scaler_X = MinMaxScaler()
            scaler_y = MinMaxScaler()
            X_test_scaled = scaler_X.fit_transform(X_test)
            y_test_scaled = scaler_y.fit_transform(y_test)
        
            # Predizioni
            self.predicted_y = model.predict(X_test_scaled)
            self.predicted_y = scaler_y.inverse_transform(self.predicted_y)
        
            # Metriche di valutazione
            mae = mean_absolute_error(y_test, self.predicted_y)
            mse = mean_squared_error(y_test, self.predicted_y)
            rmse = np.sqrt(mse)
        
            print(f"MAE: {mae}, MSE: {mse}, RMSE: {rmse}")
        
            # Plot dei risultati
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
        
            ax.scatter(y_test[:, 0], y_test[:, 1], y_test[:, 2], c='blue', label='Original Y')
            ax.scatter(self.predicted_y[:, 0], self.predicted_y[:, 1], self.predicted_y[:, 2], c='red', label='Predicted Y')
        
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.title('Confronto tra Y Originale e Y Predetto')
            plt.savefig(self.media_path + "/Predicted_points.png")
            plt.show()


