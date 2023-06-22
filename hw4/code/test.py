import numpy as np
from numpy import linalg

data = np.load('q3_1.npz')
print(data.files)
print("\nF:")
print(data['F'])

print("\nE:")
print(data['E'])