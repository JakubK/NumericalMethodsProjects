import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
from math import sqrt


def base(i, x, nodes, tab):
    result = 1
    for j in range(nodes):
        if i != j:
            result = result * ((x - tab[j]) / (tab[i] - tab[j]))
    return result

def F(x,nodes,tab,y):
    result = 0
    for i in range(nodes):
        result = result + (y[i] * base(i,x,nodes,tab))
    return result

def spaced(x, nodes):
    step = int(len(x) / nodes)
    result = [0] * nodes
    for i in range(nodes):
        result[i] = x[i * step]
    return result

def randomly_spaced(x,y, nodes):
    newX = [0] * nodes
    newY = [0] * nodes
    step = int(len(x) / nodes)

    # indices = random.sample(range(0,512),nodes)

    for i in range(nodes):
        ind = random.randint(i * step, (i+1) * step)
        newX[i] = x[ind]
        newY[i] = y[ind]

    return newX, newY

def parameters(x, y):
 
    N = 4 * (len(x) - 1)
    A = np.zeros((N, N))
    b = np.zeros((N, 1))
 
    for i in range(len(x) - 1):
        h = x[i + 1] - x[i]
 
        # Sj(xj) = f(xj)
        A[4 * i][4 * i] = 1
        b[4 * i] = y[i]
 
        # Sj(xj+1) = f(xj+1)
        A[4 * i + 1][4 * i] = 1
        A[4 * i + 1][4 * i + 1] = h
        A[4 * i + 1][4 * i + 2] = h ** 2
        A[4 * i + 1][4 * i + 3] = h ** 3
        b[4 * i + 1] = y[i + 1]
 
        # Sj'(xj-1) = Sj'(xj)
        A[4 * i + 2][4 * (i - 1) + 1] = 1
        A[4 * i + 2][4 * (i - 1) + 2] = 2 * h
        A[4 * i + 2][4 * (i - 1) + 3] = 3 * (h ** 2)
        A[4 * i + 2][4 * i + 1] = -1
        b[4 * i + 2] = 0
 
        # Sj''(xj-1) = Sj''(xj)
        A[4 * i + 3][4 * (i - 1) + 2] = 2
        A[4 * i + 3][4 * (i - 1) + 3] = 6 * h
        A[4 * i + 3][4 * i + 2] = -2
        b[4 * i + 3] = 0
 
    # S0''(x0) = 0 and Sn-1''(xn) = 0
    A[2][2] = 1
    b[2] = 0
 
    h = x[len(x) - 1] - x[len(x) - 2]
    A[3][4 * (len(x) - 2) + 2] = 2
    A[3][4 * (len(x) - 2) + 3] = 6 * h
    b[3] = 0
 
    return np.linalg.solve(A,b)
 
 
def splineFunction(x, y):
 
    p = parameters(x, y)
 
    def fun(x0):
        for i in range(0, len(x) - 1):
            if x[i] <= x0 <= x[i + 1]:
                h = x0 - x[i]
                a = p[4 * i]
                b = p[4 * i + 1]
                c = p[4 * i + 2]
                d = p[4 * i + 3]
                return d * (h ** 3) + c * (h ** 2) + b * h + a
    return fun

fileNames = ['MountEverest.csv', 'SpacerniakGdansk.csv', 'WielkiKanionKolorado.csv']
# fileNames = ['WielkiKanionKolorado.csv']

for name in fileNames:
    data = pd.read_csv(name) #512
    x = data['Dystans (m)'].values
    y = data['Wysokość (m)'].values
    length = len(x)

    nodes = 32
    # nodes_x = spaced(x,nodes)
    # nodes_y = spaced(y,nodes)
    nodes_x, nodes_y = randomly_spaced(x,y,nodes)

    aprox = [0] * length
    for i in range(length):
        aprox[i] = F(x[i],nodes,nodes_x,nodes_y)

    plt.semilogy(x,y)
    plt.title("Interpolacja metodą wielomianu lagrange'a")
    plt.xlabel("Dystans (m)")
    plt.ylabel("Wysokość (m)")

    plt.semilogy(x,aprox)
    plt.legend(['Dokładne dane', 'Aproksymacja'])

    plt.show()

    plt.semilogy(x,y)
    plt.title("Interpolacja metodą splajnów 3 stopnia")
    plt.xlabel("Dystans (m)")
    plt.ylabel("Wysokość (m)")

    for i in range(length):
        aprox[i] = splineFunction(nodes_x,nodes_y)(x[i])

    plt.semilogy(x,aprox)
    plt.legend(['Dokładne dane', 'Aproksymacja'])
    plt.show()