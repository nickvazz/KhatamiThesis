
import matplotlib as mpl
from sys import platform
if platform == 'darwin':
    mpl.use('TkAgg')
elif platform == "linux2":
    mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from scipy.linalg import block_diag
A = [[1, 0],
     [0, 1]]
B = [[3, 4, 5],
     [6, 7, 8]]
C = [[7]]

adj_matrix = block_diag(A,B,C,A,B,A,B)



rows = set()
blocks = []
for col in range(adj_matrix.shape[1]):
    row = np.argmax(adj_matrix[:,col])
    if row not in rows:
        rows.add(row)
        blocks.append((row, col))

print (blocks)
print (len(blocks))
plt.imshow(adj_matrix)
plt.show()
