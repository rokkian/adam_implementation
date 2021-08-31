from numpy import asarray, arange, meshgrid
import numpy as np
from matplotlib import pyplot
from space_3d import objective

def plot2d(bounds:np.array, function, solutions:list=list()):
    '''A partire da un array che definisce lo spazio, e la funzione da usare, plotta lo spazio 2d'''
    #campiono lo spazio dato dal range accettato
    xaxis=arange(bounds[0, 0], bounds[0, 1], 0.1)
    yaxis=arange(bounds[1, 0], bounds[1, 1], 0.1)

    #creo il reticolo
    x, y = meshgrid(xaxis, yaxis)

    #target
    results = function(x, y)

    #plot in 2d, con 50 livelli e jet color
    pyplot.contourf(x, y, results, levels=50, cmap='jet')
    if solutions:
        solutions = asarray(solutions)
        pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w')
    pyplot.show()

if __name__ == '__main__':
    #bounds
    bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
    print("Bounds e': ", bounds)

    plot2d(bounds, objective)