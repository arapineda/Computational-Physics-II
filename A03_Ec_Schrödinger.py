import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig
from scipy.linalg import expm

def dermat(size, prm) :
    '''
    Parámetros
    ----------
    size : dimensión de la matriz
    prm : determina qué tipo de matriz se genera -- 1 en diagonal superior o -1
    en la diagonal inferior a la principal

    Regresa
    -------
    Obtiene una de dos posibles matrices con valores de -1 y 1 en las 
    diagonales adyacentes a la principal.
    '''
    mat = np.zeros((size,size))
    np.fill_diagonal(mat, 1)
    if prm == 1 :
        mat = np.column_stack((np.zeros(size), mat))
        mat = np.row_stack((mat, np.zeros(size + 1)))
    elif prm == -1 :
        mat = np.row_stack((np.zeros(size), mat))
        mat = np.column_stack((mat, np.zeros(size + 1)))
    return mat
                              
def matfin(size, h) :    
    '''
    Parámetros
    ----------
    size : dimensión de la matriz
    h : separación entre puntos

    Regresa
    -------
    Obtiene el operador de segunda derivada en forma de matriz.
    '''    
    mat1 = dermat(size, 1)
    mat2 = dermat(size, -1)
    dmat = (mat1 + mat2)
    np.fill_diagonal(dmat, -2)
    dmat = 1/(h**2) * dmat 
    return dmat

def hamilton(N,x) :
    '''
    Parámetros
    ----------
    n : dimensión de la matriz
    x : vector de posición

    Regresa
    -------
    Operador del Hamiltoniano y diccionario con eigenvalores y eigenfunciones 
    en orden ascendente de eigenvalores.
    '''
    h = x[2] - x[1]
    dmat = matfin(N-1,h)
    T = -0.5 * dmat
    U = 0.5 * np.identity(N) * x**2
   #U = 0 
    H = T + U
    eigvalue, eigvector = eig(H)
    eigvalue2 = np.sort(eigvalue)
    #eigdict = dict(zip(eigvalue,eigvector))

    #sorted(eigdict.items())
    return H, eigvalue, eigvalue2, eigvector

    
def main() :
    
    a = 4
    N = 1024
    t = 100
    h = (2*a)/(N-1)    
    x = np.linspace(-a, a, N)
    
    H, eigvalue, eigvalue2, eigvector = hamilton(N, x)
    
    eigvalue = list(eigvalue)
    eigvalue2 = list(eigvalue2)
    
    indeig1 = eigvalue.index(eigvalue2[0])
    indeig2 = eigvalue.index(eigvalue2[1])

    psi = eigvector[:, indeig1] + eigvector[:, indeig2]
    
    opexp = expm(-1j * H * t)
    
    evol = np.matmul(opexp, psi)
    
    denpro = (np.absolute(evol))**2
    
    plt.plot(x,denpro)
    plt.xlabel('x')
    plt.ylabel('Densidad de probabilidad')
    plt.title('Caso del oscilador armónico')
    plt.show()

    return None

main()
