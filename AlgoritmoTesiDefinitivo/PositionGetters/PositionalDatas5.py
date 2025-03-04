import numpy as np
from PositionGetters.PositionalData import PositionalData 
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error

class PositionalDatas5(PositionalData):
    def __init__(self,timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test, magnetometerData, gyroscope, pressure, ELA, ifft, PD1, PD3):
        super().__init__(timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test,specificPD="PD5") 
        self.Mag = magnetometerData
        self.Gyro = gyroscope
        self.Press = pressure
        self.ELA = ELA
        self.ifft = ifft
        self.PD1 = PD1
        self.PD3 = PD3
        

    def processData(self):
        self.timestamp = self.timestamp.reshape(-1, 1)
        dati = np.concatenate((self.timestamp, self.Acc, self.Gyro, self.Orient, self.Mag, self.Press, self.ELA, self.ifft, self.PD1, self.PD3), axis=1)
        nn = self.NeuralNetwork(dati, self.file_manager.mediaDir, self.file_index, self.directory)
        #nn.train_model()
        mae, distanza = nn.predict_new_data()
        PositionDataFrame = pd.DataFrame(nn.predicted_y)
        self.file_manager.save_position_data(PositionDataFrame, "PositionalData5")
        print(f"Tempo impiegato={self.timestamp[-1]}")
        
        return distanza
       
    class NeuralNetwork:
        def __init__(self, test_X_path, media_path, file_index, directory):
            self.path_points=f"{directory}/Training"
            self.path_data=f"{directory}/File_uniti"
            self.X_paths = [f"{self.path_data}/{file_index}.csv"]  # Lista dei path per X
            self.y_paths = [f"{self.path_points}/p_{file_index}_reconstructed.csv"]  # Lista dei path per Y
            self.model_path = f"{directory}/Training/trained_model_5.h5"  # Percorso per salvare il modello
            self.test_X_path = test_X_path  # Percorso del dataset di test X
            self.media_path = media_path
    
        def load_data(self, paths):
            all_data = []
            for path in paths:
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
            X = self.test_X_path
            self.y = self.load_data(self.y_paths)
        
            # Preprocessing
            X_scaled, y_scaled = self.preprocess_data(X, self.y)
        
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
            model = keras.models.load_model(self.model_path)
        
            # Carica il dataset di test
            X_test = self.test_X_path
        
            # Normalizzazione
            scaler_X = RobustScaler()
            X_test_scaled = scaler_X.fit_transform(X_test)
        
            # Predizioni
            self.predicted_y = model.predict(X_test_scaled)
            self.predicted_y[self.predicted_y[:, 2] < 0.1, 2] = 0
            
            # Calcolo metriche
            # mae = mean_absolute_error(self.y, self.predicted_y)

            # Calcola la differenza tra punti consecutivi
            differenze = np.diff(self.predicted_y, axis=0)
    
            # Calcola la distanza euclidea tra ogni coppia di punti consecutivi
            distanze = np.linalg.norm(differenze, axis=1)
    
            # Somma tutte le distanze per ottenere la distanza totale percorsa
            distanza_totale = np.sum(distanze)

            # Stampa delle metriche per debugging
            #print(f"Test Performance PD5: MAE={mae}, metri percorsi={distanza_totale}")
        
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
            #plt.show()
            
            return 0, distanza_totale


        
            
