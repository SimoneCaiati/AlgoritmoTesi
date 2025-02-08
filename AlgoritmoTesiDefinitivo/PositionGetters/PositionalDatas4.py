from turtle import position
import numpy as np
from PositionGetters.PositionalData import PositionalData 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PositionalDatas4(PositionalData):
    def __init__(self,timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test, magnetometerData, gyroscope, pressure, ELA):
        super().__init__(timestamp,accelerometerData,orientationData,sample_rate,file_index,directory,test,specificPD="PD4") 
        self.Mag = magnetometerData
        self.Gyro = gyroscope
        self.Press = pressure
        self.ELA = ELA
        

    def processData(self):
        self.linearRegression()
        y_pred_ordered= self.y_pred[np.lexsort((self.y_pred[:, 1], self.y_pred[:, 0]))]
        PositionDataFrame=pd.DataFrame(y_pred_ordered)
        self.file_manager.save_position_data(PositionDataFrame, "PositionalData4")
        self.visualizer.position=y_pred_ordered
        self.visualizer.plot_path()
        
    def linearRegression(self):
        # PD4DS = (LA, MF, AV, O, ELA, A)
        self.Press = self.Press.reshape(-1, 1)
        X = np.concatenate((self.Acc, self.Mag, self.Gyro, self.Orient, self.ELA, self.Press), axis=1)
        y = pd.read_csv(f"../AlgoritmoTesiDefinitivo/{self.directory}/PositionalDatas/{self.file_index}/Real.csv", delimiter=',', na_values=['']).replace(" ", "").to_numpy().astype(float)

        # 3. Divideo il dataset in training e test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 4. Addestro il modello di regressione lineare
        model = LinearRegression()
        model.fit(X_train, y_train)

        # 5. Predico i valori sui dati di test
        self.y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, self.y_pred)
        print(f"Errore quadratico medio (MSE): {mse}")

        # 6. Grafico 3D per visualizzare i risultati
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Dati reali
        ax.scatter(y_test[:, 0], y_test[:, 1], y_test[:, 2], color='blue', label='Valori Reali')

        # Dati predetti
        ax.scatter(self.y_pred[:, 0], self.y_pred[:, 1], self.y_pred[:, 2], color='red', label='Valori Predetti')

        # Etichette degli assi
        ax.set_xlabel('Asse X')
        ax.set_ylabel('Asse Y')
        ax.set_zlabel('Asse Z')
        ax.set_title('Confronto tra Valori Reali e Predetti')

        # Legenda
        ax.legend()

        # Mostrare il grafico
        plt.savefig(self.file_manager.mediaDir + "/Predicted_points.png")
        plt.show()



