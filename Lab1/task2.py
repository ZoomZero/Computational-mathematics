import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as lg


def nonZeroElemFill(arr, num):
    for i in range(num):
        for j in range(num):
            if i != 0 and j != 0 and i != num - 1 and j != num - 1:
                nonZero = num * i + j
                arr.append(nonZero)


def matrixFill(matr, arr, num, step):
    for elem in arr:
        for i in range(num - 2):
            for j in range(num - 2):
                    elemNum = num * (i + 1) + (j + 1)
                    if elemNum == elem:
                        matr.append(int(4 / (step ** 2)))
                    elif abs(elemNum - elem) == 1 or abs(elemNum - elem) == num:
                        matr.append(int(- 1 / (step ** 2)))
                    else:
                        matr.append(0)


def rowFill(arr, center):
    if (center % 2 == 1):
        row[center // 2] = 1
    else:
        row[center // 2] = 1
        row[center // 2 + 1] = 1
        row[center // 2 - 1] = 1
        row[center // 2 - 2] = 1


def error(matrix, res, row):
    n = len(row)
    Ax = np.dot(matrix, res)
    normAxF = 0
    normF = 0
    for i in range(n):
        normAxF += (Ax[i] - row[i]) ** 2
        normF += row[i] ** 2

    return np.sqrt(normAxF) / np.sqrt(normF)


def seidel(matr, row, eps):
    n = len(row)
    x = np.zeros(n)
    errArray = []

    while True:
        err = error(matr, x, row)
        errArray.append(err)

        if err >= eps:
            x_new = np.copy(x)
            for j in range(n):
                d = row[j]
                for i in range(n):
                    if(j != i):
                        d -= matr[j][i] * x[i]
                x_new[j] = d / matr[j][j]
            x = x_new
        else:
            break

    return errArray


def jacobi(matr, row, eps):
    n = len(row)
    x = np.zeros(n)
    errArray = []
    D = np.diag(matr)
    R = matr - np.diagflat(D)

    while True:
        err = error(matr, x, row)
        errArray.append(err)

        if err >= eps:
            for i in range(n):
                x = (row - np.dot(R, x)) / D
        else:
            break

    return errArray


def mpiT(matr, row, eps):
    n = len(row)
    x = np.zeros(n)
    lmbd = np.linalg.eigvals(matr)
    t = 2 / (max(lmbd) + min(lmbd))
    errArray = []
    g = np.dot(t, row)
    E = np.eye(n)
    R = E - np.dot(t, matr)

    while True:
        err = error(matr, x, row)
        errArray.append(err)

        if err >= eps:
            x = np.dot(R, x) + g
        else:
            break

    return errArray


def cg(matr, row, eps):
    n = len(row)
    x = np.zeros(n)
    errArray = []

    while True:
        err = error(matr, x, row)
        errArray.append(err)
        if err >= eps:
            x_new = np.copy(x)
            x_new = lg.cg(matr, row, x, maxiter=1)
            x = x_new[0]
        else:
            break

    return errArray


print("Input h")
h = float(input())

stepNumber = int(1 / h) + 1

nonBorderNum = stepNumber ** 2 - 4 * (stepNumber - 1)

nonZero = []
nonZeroElemFill(nonZero, stepNumber)

matrix = []
matrixFill(matrix, nonZero, stepNumber, h)
matrix = np.array(matrix).reshape(nonBorderNum, nonBorderNum)

row = [0 for i in range(nonBorderNum)]
rowFill(row, nonBorderNum)

res = np.linalg.solve(matrix, row)
print("Solution is:", end=' ')
print(res)

e = 1e-3

errJacobi = jacobi(matrix, row, e)
errSeidel = seidel(matrix, row, e)
errMpiT = mpiT(matrix, row, e)
errCG = cg(matrix, row, e)

fig, ax = plt.subplots(num=1)
plt.title("Matrix visualization")
plt.xlabel('X')
plt.ylabel('Y')
pic = ax.imshow(matrix, cmap=plt.get_cmap('jet'), aspect='equal',
                interpolation='lanczos')
fig.colorbar(pic, ax=ax)
plt.savefig("Matrix Vizualization 2.png")

fig2, ax2 = plt.subplots(num=2)
resSide = int(np.sqrt(nonBorderNum))
resultMatrix = res.reshape(resSide, resSide)
plt.title("Soluion visualization")
plt.xlabel('X')
plt.ylabel('Y')
pic = ax2.imshow(resultMatrix, cmap=plt.get_cmap('jet'), aspect='equal',
                 interpolation='lanczos')
fig2.colorbar(pic, ax=ax2)
plt.savefig("Solution vizualization.png")

fig3 = plt.figure(num=3)
plt.title("Solution convergence")
plt.xlabel('Number of iteration')
plt.ylabel('||Ax_n - f|| / ||f||')
itNumJacobi = np.arange(0, len(errJacobi))
itNumSeidel = np.arange(0, len(errSeidel))
itNumMPI = np.arange(0, len(errMpiT))
itNumCG = np.arange(0, len(errCG))
plt.plot(itNumJacobi, errJacobi, color='green')
plt.plot(itNumSeidel, errSeidel, color='red')
plt.plot(itNumMPI, errMpiT, color='blue')
plt.plot(itNumCG, errCG, color='black')
plt.savefig("Solution convergence.png")

plt.show()
