# Perceptron Implementation in Python - Bases
# Sumber: Matematika Untuk DL (M. Syamsuddin, 2021)

import numpy as np

# Y : Sig(XW)
# X dim : (N,M)
# Wt dim : (1, M)
# Z = XW dim : (N, M)
# N : 1..t..N Number of samples
# M : Number of features

def sigmoid(Zt : np.ndarray):
    # Fungsi Sigmoid
    # Zt: Z ke-t, ndarray NxM
    return 1/np.exp(-Zt)


def random_arr(N : int, M : int) -> np.ndarray:
    # N, M: int
    rng = np.random.default_rng()
    return rng.random((N, M))


class perceptron:
    # Minsky's Single Perceptron
    # with Sigmoid Function
    # only 1 output per t input
    # atributes
    def __init__(self, M : int):
        self.M = M
        self.X = None
        self.W = random_arr(M, 1)
        self.Y = None
        # self.alpha = sigmoid function
    # Setter
    def set_M(self, M : int):
        self.M = M
        self.W = random_arr(M, 1)
        print("Weights are reset")
    def set_X(self, X : np.ndarray):
        if X.shape[1] != self.M:
            raise ValueError("Number of features did not match!")
        self.X = X
    def set_W(self, W : np.ndarray):
        self.W = W
    # Getter 
    def get_M(self) -> int:
        return self.M
    def get_X(self) -> np.ndarray:
        return self.X
    def get_W(self) -> np.ndarray:
        return self.W
    # Z
    def 

