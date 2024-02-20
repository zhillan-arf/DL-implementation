# 18220008 zhil
# Perceptron Implementation in Python - Bases
import numpy as np

# Math
def sigmoid(zt : float) -> float:
    return 1/np.exp(zt)
def loss(y, yh):
    return 0.5 * (y - yh)**2

# Utils
def random_arr(N : int, M : int) -> np.ndarray:
    # N, M: int
    rng = np.random.default_rng()
    return rng.random((N, M))

class SPM:  # Single Perceptron Model
    def __init__(self):
        self.M = self.w = self.Z = self.Yh = self.e = self.C = None
        # self.activation_function = sigmoid
    
    def forward_prop(self, X : np.ndarray, Y : np.ndarray):
        N = X.shape[0]
        self.M = X.shape[1]
        self.w = random_arr(self.M, 1)
        self.Z = random_arr(N, 1)
        self.Yh = random_arr(N, 1)
        self.e = random_arr(N, 1)
        self.Z = np.dot(X, self.w)
        self.Yh = sigmoid(self.Z)
        self.e = Y - self.Yh
        self.C = np.dot(self.e.transpose(), self.e)

    # Backwards propagation methods
    def back_prop(self, X : np.ndarray, learning_rate : float):
        w1 = self.w + learning_rate * np.dot(X.transpose(), (self.e * self.Yh * (1 - self.Yh)))
        self.w = w1
    
    def train(self, X : np.ndarray, Y : np.ndarray, learning_rate : float) -> float:
        self.forward_prop(X, Y)
        self.back_prop(X, learning_rate)
    
    def show_results(self):
        print(self.C)
        print(self.Yh)

# Sumber: Matematika Untuk DL (M. Syamsuddin, 2021)
