# Test Grounds
import numpy as np
from bases import SPM

X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10,11,12]])

Y = np.array([[6],
              [5],
              [4],
              [3]])

spm1 = SPM()
spm1.train(X, Y, 0.1)
spm1.show_results()