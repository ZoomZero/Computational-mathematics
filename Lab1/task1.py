import matplotlib.pyplot as plt
import numpy as np


def matrixFill(line, column, mat):
    mat[line][column] = mat[line][column] / (h ** 2)


h = 0.25
matrix = np.array([[4, -1, 0, -1, 0, 0, 0, 0, 0],
                   [-1, 4, -1, 0, -1, 0, 0, 0, 0],
                   [0, -1, 4, 0, 0, -1, 0, 0, 0],
                   [-1, 0, 0, 4, -1, 0, -1, 0, 0],
                   [0, -1, 0, -1, 4, -1, 0, -1, 0],
                   [0, 0, -1, 0, -1, 4, 0, 0, -1],
                   [0, 0, 0, -1, 0, 0, 4, -1, 0],
                   [0, 0, 0, 0, -1, 0, -1, 4, -1],
                   [0, 0, 0, 0, 0, -1, 0, -1, 4]])

for i in range(9):
    for j in range(9):
        matrixFill(i, j, matrix)

row = np.ones(9)

print("Matrix is:")
print(matrix)

fig, ax = plt.subplots()
plt.title("Matrix visualization")
plt.xlabel('X')
plt.ylabel('Y')
pic = ax.imshow(matrix, cmap=plt.get_cmap('jet'), aspect='equal', interpolation='lanczos')
fig.colorbar(pic, ax=ax)
plt.savefig("Matrix Vizualization 1.png")

res = np.linalg.solve(matrix, row)
print("Solution is:", end=' ')
print(res)

plt.show()
