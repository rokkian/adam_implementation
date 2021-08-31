from numpy import arange, meshgrid
from matplotlib import pyplot

#funzione obiettivo
def objective(x, y):
    '''funzione obiettivo z = x**2 + y**2 '''
    return x ** 2.0 + y ** 2.0

if __name__ == '__main__':
    #range dell'input
    r_min, r_max = -1.0, 1.0

    #campiono il range in modo uniforme
    xaxis = arange(r_min, r_max, 0.1)
    yaxis = arange(r_min, r_max, 0.1)

    #creo un mesh dall'asse
    x, y = meshgrid(xaxis, yaxis, sparse=False)
    print(x,'\n',y)

    #calcolo i target
    results = objective(x, y)

    #creo il plot della superficie, con il color scheme jet
    figure = pyplot.figure()
    axis = figure.gca(projection='3d')
    axis.plot_surface(x, y, results, cmap="jet")
    pyplot.show()