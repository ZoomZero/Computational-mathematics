import matplotlib.pyplot as plt
import numpy as np

# для дебага
def print_m(matrix):
    for row in matrix:
        print(' '.join(str(col) for col in row))
        
def print_a(array):
    for elem in array:
        print(''.join(str(elem)), end=' ')
    print()


h = 0.05
start = -1
end = 1
u0 = 0 # граничные условия


def f(x, y):
    if (x > -0.7 and x < -0.1 and y > 0.1 and y < 0.7):
        return 10
    else:
        return 0
    

def a(x):
    return (x**2 + 1) / 10


def a_(x):
    return x / 5


def x_i(i) :
    return -1.0 + i * h

def y_j(j) :
    return -1.0 + j * h


def main():
    # количество шагов по осям
    n = int((end - start) / h + 1)
    # шаг где заходим за угол
    corn = int(n/2) - 1

    x = np.linspace(start, end, n)
    y = np.linspace(start, end, n)
    X, Y = np.meshgrid(x, y, indexing='ij')
    # правая часть уравнения ненулевая в квадрате
    b = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            b[i][j] = f(x_i(i), y_j(j))
    # print_m(b)  
    
    
    u = np.empty((n, n))
    # считаем начальным приближением 0, тем самым задаем нулевые граничные условия
    unew = np.zeros((n, n))
    # разница между шагами алгоритма для точности
    l2_diff = 1.0
    eps = 10**(-6)
    tol_hist_gs = []
    it = 0
    
    while (l2_diff > eps):
        np.copyto(u, unew)
        for j in range(1, n-1):
            for i in range(1, n-1):
                # игнорируем угол, в котором нет области
                if (i > corn and j < corn):
                    continue
                else:
                    unew[i, j] = ((a_(x_i(i)) / (2*h) + a(x_i(i)) / (h**2)) * unew[i-1, j] + \
                                  (-a_(x_i(i)) / (2*h) + a(x_i(i)) / (h**2)) * u[i+1, j] + \
                                  (a(x_i(i)) / (h**2)) * unew[i, j-1] + \
                                  (a(x_i(i)) / (h**2)) * u[i, j+1] + b[i, j]) / (4*a(x_i(i)) / (h**2))
        # l2_diff = np.sum(np.power((unew-u),2))
        l2_diff = (np.linalg.norm(unew-u,ord=2) / np.linalg.norm(unew,ord=2))
        tol_hist_gs.append(l2_diff)
        it += 1
    
    # print('Количество итераций: %d' % it)
    fig, (ax_1, ax_2) = plt.subplots(1, 2, figsize=(16,5))
    pic = ax_1.contourf(X, Y, unew, 10)
    # pic = ax_1.imshow(X, Y, unew, cmap=plt.get_cmap('jet'), aspect='equal', interpolation='lanczos')
    fig.colorbar(pic, ax=ax_1)
    ax_1.set_xlabel(r'$x$')
    ax_1.set_ylabel(r'$y$')
    ax_1.set_title('Решение')
    # ax_2.semilogy(tol_hist_gs, color='red', label='Gauss-Seidel')
    ax_2.plot(tol_hist_gs, color='red', label='Gauss-Seidel')
    ax_2.set_xlabel('iteration')
    ax_2.set_ylabel('l2_diff')
    ax_2.set_title('Сходимость')
    ax_2.legend()
            
    
if __name__ == '__main__':
    main()