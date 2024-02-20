# Test Grounds
import numpy as np

# def sigmoid(zt : float) -> float:
#     return 1/np.exp(zt)

# print(sigmoid(6))

from bases import sigmoid


X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10,11,12]])

Y = np.array([[6],
              [5],
              [4],
              [3]])


N = X.shape[0]
M = X.shape[1]
# print(N, M)

# from bases import random_arr
# w = random_arr(M, 1)
# Z = random_arr(N, 1)
# Yh = random_arr(N, 1)
# e = random_arr(N, 1)

# w2 = w
# Z2 = Z
# Yh2 = Yh
# e2 = e

# for i in range(0, N):
#     Z[i] = np.dot(X[i], w)
#     Yh[i] = sigmoid(Z[i])
#     e[i] = Y[i] - Yh[i]

# print("New:\n\n", w, "\n\n", Z, "\n\n", Yh, "\n\n", e, "\n\n")

# Z2 = np.dot(X, w2)
# Yh2 = sigmoid(Z)
# e2 = Y - Yh2

# print("New 2 vectorized:\n\n", w, "\n\n", Z, "\n\n", Yh, "\n\n", e, "\n\n")

# C = np.dot(e.transpose(), e)

# print(C)



X2 = np.array([[1, 2, 3],
               [4, 5, 6],
               [7, 8, 9],
               [10,11,12]])

# print(X * X2)

from bases import SPM

spm1 = SPM()
spm1.train(X, Y, 0.1)
spm1.show_results()