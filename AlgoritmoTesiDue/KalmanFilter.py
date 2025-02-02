import numpy as np

class KalmanFilter(object):
    def __init__(self, F = None, B = None, H = None, Q = None, R = None, P = None, x0 = None):

        if(F is None or H is None):
            raise ValueError("Set proper system dynamics.")

        self.n = F.shape[1]
        self.m = H.shape[1]

        self.F = F
        self.H = H
        self.B = 0 if B is None else B
        self.Q = np.eye(self.n) if Q is None else Q
        self.R = np.eye(self.n) if R is None else R
        self.P = np.eye(self.n) if P is None else P
        self.x = np.zeros((self.n, 1)) if x0 is None else x0
        self.I = np.eye(self.n)

    def predict(self, u):
        """Step di predizione"""
        self.x = np.dot(self.F, self.x) + np.dot(self.B, u)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.x

    def update(self, z):
        """"Step di aggiornamento"""
        y = z - np.dot(self.H, self.x)
        S = self.R + np.dot(np.dot(self.H,self.P) , self.H.T)
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.x = self.x + np.dot(self.R, y)
        self.P = np.dot(self.I - np.dot(K, self.H), self.P)
        	
    def get_state(self):
        """Ritorna lo stato corrente."""
        return self.x
    


