import numpy as np


class Kalman:
    def __init__(self, initial_state, dt: float = 1.0):
        self.dt = dt
        self.F = np.array([[1, 0, dt, 0],
                           [0, 1, 0, dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]], dtype=float)
        self.H = np.array([[1, 0, 0, 0],
                           [0, 1, 0, 0]], dtype=float)
        q = 0.8
        dt2 = dt ** 2
        self.Q = np.array([[dt2*dt2/4, 0, dt2*dt/2, 0],
                           [0, dt2*dt2/4, 0, dt2*dt/2],
                           [dt2*dt/2, 0, dt2, 0],
                           [0, dt2*dt/2, 0, dt2]], dtype=float) * q
        self.R = np.eye(2) * 0.7
        self.x = np.array(initial_state).reshape(4, 1)
        self.P = np.eye(4) * 50

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x.flatten()

    def update(self, z):
        z = np.array(z).reshape(2, 1)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self.H) @ self.P