import numpy as np
from matplotlib import pyplot
from math import sqrt
from numpy import asarray, arange, meshgrid
from numpy.random import rand, seed
from mpl_toolkits.mplot3d import Axes3D
from space_3d import objective
from space_2d import plot2d


def derivative(x, y):
    '''Derivata della funzione obiettivo, z'=2x + 2y'''
    return asarray([x * 2.0, y*2.0])

def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
    """Adam alg"""
    solutions = list()

    #genero un punto iniziale
    x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])    #numero casuale compreso tra il minimo e il max dello spazio
    score = objective(x[0], x[1])

    #inizializzo il primo e il secondo momoento
    m = [0.0 for _ in range(bounds.shape[0])]
    v = [0.0 for _ in range(bounds.shape[0])]

    # avvio gli updates del gradient descent
    for t in range(n_iter):
        #calcolo il gradiente
        g = derivative(x[0], x[1])

        #costruisco la soluzione una variabile alla volta
        for i in range(bounds.shape[0]):
            # m(t) = beta1 * m(t-1) + (1-beta1) * g(t)
            m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
            
            # v(t) = beta2 * v(t-1) + (1-beta2) * g(t)^2
            v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] ** 2

            # mhat(t) = m[i] / (1 - beta1(t))
            mhat = m[i] / (1 - beta1 ** (t+1))

            # vhat(t) = v[i] / (1 - beta2(t))
            vhat = v[i] / (1 - beta2 ** (t+1))

            # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + ep)
            x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
        
        # valuta il candidate point
        score = objective(x[0], x[1])

        #tieni traccia delle soluzioni
        solutions.append(x.copy())

        print(f'>{t} f({str(x)}) = {score:.5f}')
    
    return solutions

if __name__ == '__main__':
    seed(2)     #imposto il generatore random
    bounds = asarray([[-1., 2.], [-1., 1.]])
    n_iter = 600         # totale di iterazioni
    alpha = 0.02        # steps size
    beta1 = 0.8         # fattore per il gradiente medio
    beta2 = 0.999       # fattore per il gradiente quadrato medio
    
    solutions = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)

    #plot
    plot2d(bounds, objective, solutions)
