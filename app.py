# Test Grounds
import numpy as np

X = np.array([[1,2,3],
              [4,5,6],
              [7,8,9],
              [10,11,12]])

M = 3

# print(X.shape[1])

rng = np.random.default_rng()
a = 5 * rng.random((3, 2)) - 5
# print(a)

from bases import random_arr
# print(random_arr(4,3))

from bases import perceptron
p1 = perceptron(3)
# print(p1.get_M(), p1.get_X(), p1.get_W())