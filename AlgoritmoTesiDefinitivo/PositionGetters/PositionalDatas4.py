import numpy as np
import os
from PositionGetters.PositionalData import PositionalData 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import RobustScaler

class PositionalDatas4(PositionalData):
    def __init__(self,timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test, magnetometerData, gyroscope, pressure, ELA):
        super().__init__(timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test,specificPD="PD4") 
        self.Mag = magnetometerData
        self.Gyro = gyroscope
        self.Press = pressure
        self.ELA = ELA
        self.test_path="__empty__"
        

    def processData(self):
        self.timestamp = self.timestamp.reshape(-1, 1)
        dati = np.concatenate((self.timestamp, self.Acc, self.Gyro, self.Orient, self.Mag, self.Press), axis=1)
        if self.test == False:
            nn = self.NeuralNetwork(dati, self.file_manager.mediaDir, self.file_index)
        else:
            nn = self.NeuralNetwork(dati, self.file_manager.mediaDir, self.file_index)
            nn.y_paths=self.test_path
        nn.train_model()
        nn.predict_new_data()
        PositionDataFrame = pd.DataFrame(nn.predicted_y)
        self.file_manager.save_position_data(PositionDataFrame, "PositionalData4")
       
    class NeuralNetwork:
        def __init__(self, test_X_path, media_path, file_index):
            self.path_points="SensorLogger/Training"
            self.path_data="SensorLogger/File_uniti"
            self.X_paths = [f"{self.path_data}/{file_index}.csv"]  # Lista dei path per X
            self.y_paths = [f"{self.path_points}/p_{file_index}_reconstructed.csv"]  # Lista dei path per Y
            self.model_path = "SensorLogger/Training/trained_model.h5"  # Percorso per salvare il modello
            self.test_X_path = test_X_path  # Percorso del dataset di test X
            self.media_path = media_path
    
        def load_data(self, paths):
            all_data = []
            for path in paths:
                print(f"Caricamento file: {path}")  # DEBUG: stampa il percorso
                if not os.path.exists(path):
                    raise FileNotFoundError(f"File non trovato: {path}")
                data = pd.read_csv(path, delimiter=',', na_values=['']).replace(" ", "").dropna().to_numpy().astype(float)
                all_data.append(data)
            return np.vstack(all_data)
    
        def preprocess_data(self, X, y):
            scaler_X = RobustScaler()
            scaler_y = RobustScaler()

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
            model.fit(X_scaled, y_scaled, epochs=50, batch_size=32, callbacks=[early_stopping, reduce_lr])
        
            # Salvataggio del modello
            model.save(self.model_path)
            print(f"Modello salvato in {self.model_path}")
    
        def predict_new_data(self):
            # Carica il modello addestrato
            keras.utils.get_custom_objects().update({"mse": keras.losses.MeanSquaredError()})
            model = keras.models.load_model(self.model_path)
        
            # Carica il dataset di test
            X_test = self.test_X_path
        
            # Normalizzazione
            scaler_X = RobustScaler()
            X_test_scaled = scaler_X.fit_transform(X_test)
        
            # Predizioni
            self.predicted_y = model.predict(X_test_scaled)
            self.predicted_y[self.predicted_y[:, 2] < 0.001, 2] = 0
        
            # Plot dei risultati
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
        
            ax.scatter(self.predicted_y[:, 0], self.predicted_y[:, 1], self.predicted_y[:, 2], c='red', label='Predicted Y')
        
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.legend()
            plt.title('Punti Predetti')
            plt.savefig(self.media_path + "/Predicted_points.png")
            plt.show()
            


