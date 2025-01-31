import numpy as np
from scipy.fft import ifft
from scipy.linalg import inv

class KalmanFilter:
    def __init__(self, F, H, Q, R, P, x0):
        """
        Inizializza il filtro di Kalman.
        :param F: Matrice di transizione di stato.
        :param H: Matrice di osservazione.
        :param Q: Matrice di rumore di processo.
        :param R: Matrice di rumore di osservazione.
        :param P: Matrice di covarianza iniziale.
        :param x0: Stato iniziale.
        """
        self.F = F  # Matrice di transizione
        self.H = H  # Matrice di osservazione
        self.Q = Q  # Rumore di processo
        self.R = R  # Rumore di osservazione
        self.P = P  # Covarianza
        self.x = x0  # Stato iniziale

    def predict(self):
        """Fase di previsione."""
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q

    def update(self, z):
        """Fase di aggiornamento."""
        y = z - np.dot(self.H, self.x)  # Residuo
        S = np.dot(self.H, np.dot(self.P, self.H.T)) + self.R
        K = np.dot(np.dot(self.P, self.H.T), inv(S))  # Guadagno di Kalman
        self.x = self.x + np.dot(K, y)
        self.P = self.P - np.dot(K, np.dot(self.H, self.P))

    def get_state(self):
        """Ritorna lo stato corrente."""
        return self.x
    


