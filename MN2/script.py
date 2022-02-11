import numpy as np
import math
import matplotlib.pyplot as plt

import time

c = 4
d = 3
e = 0
f = 0

N = 9 * 100 +  c * 10 + d
A = np.zeros((N,N))
a1 = 5 + e
a2 = a3 = -1
#A
def setupA(N,a1,a2,a3):
    A = np.zeros((N,N))
    for i in range(0,N):
        A[i][i] = a1
    for i in range(0,N-1):
        A[i+1][i] = a2
    for i in range(0,N-2):
        A[i+2][i] = a3
    for i in range(0,N-1):
        A[i][i+1] = a2
    for i in range(0,N-2):
        A[i][i+2] = a3
    return A
A = setupA(N,a1,a2,a3)

b = np.zeros((N,1))
for i in range(0,N):
    b[i][0] = math.sin((i+1) * (f + 1))
#B

def norm(vector, N):
    nor = 0
    for i in range(0,N):
        nor = nor + (vector[i][0] ** 2)
    return math.sqrt(nor)

def jacobi(A, N):
    r = res = prevR = np.ones((N,1))
    resNorm = 1
    epsilon = 10**(-9)
    iteration = 0

    while resNorm > epsilon:
        for i in range(0, N):
            sum = 0
            for j in range(0, N):
                if j != i:
                    sum = sum + A[i][j] * prevR[j][0]
            r[i][0] = (b[i][0] - sum) / A[i][i]

        prevR = np.copy(r)
        res = np.subtract(np.dot(A,r),b)
        resNorm = norm(res,N)
        # print(resNorm)
        if np.isnan(resNorm):
            print("Jacobi diverges")
            break
        iteration = iteration + 1
    print("Norma " + str(resNorm))
    print("Iteracje " + str(iteration))

def gauss(A, N):
    r = res = prevR = np.ones((N,1))
    resNorm = 1
    epsilon = 10**(-9)
    iteration = 0

    while resNorm > epsilon:
        for i in range(0, N):
            sum = 0
            for j in range(0, i):
                sum = sum + A[i][j] * r[j][0]
            for j in range(i+1, N):
                sum = sum + A[i][j] * prevR[j][0]
            r[i][0] = (b[i][0] - sum) / A[i][i]
        prevR = np.copy(r)
        res = np.subtract(np.dot(A,r),b)
        resNorm = norm(res,N)
        # print(resNorm)
        if np.isnan(resNorm):
            print("Gauss diverges")
            break
        iteration = iteration + 1
    print("Norma " + str(resNorm))
    print("Iteracje " + str(iteration))

# start = time.time()
# jacobi(A,N)
# end = time.time()
# print("Jacobi : " + str((end - start)))
# start = time.time()
# gauss(A,N)
# end = time.time()
# print("Gauss-Seidl : " + str((end - start)))
        
#C
a1 = 3
a2 = -1
a3 = -1

N = 9 * 100 +  c * 10 + d
A = setupA(N,a1,a2,a3)

# jacobi(A,N)
# gauss(A,N)

#D

def LU(A,b, N):
    U = np.copy(A)
    L = np.identity(N)
    print('xd')
    for k in range(N-1):
        for j in range(k+1, N):
            L[j][k] = U[j][k] / U[k][k]
            # U[j][k:N] = np.subtract(U[j][k:N], L[j][k] * U[k][k:N])
            for u in range(k,N):
                U[j][u] = U[j][u] - (L[j][k] * U[k][u])


    x = np.ones((N,1))
    y = np.zeros((N,1))
    for i in range(0,N):
        value = b[i][0]
        for j in range(0,i):
            value = value - L[i][j] * y[j][0]
        y[i][0] = value / L[i][i]

    for i in range(N-1,-1,-1):
        value = y[i][0]
        for j in range(i+1,N):
            value = value - U[i][j] * x[j][0]
        x[i][0] = value / U[i][i]
    
    res = np.subtract(np.dot(A,x),b)
    print("Norma: " + str(norm(res,N)))

LU(A,b,N)

#E
a1 = 5 + e
a2 = a3 = -1
# N = [100,500,1000,2000,3000]
N = [100,200,500,800,1000]


J = [0] * len(N)
G = [0] * len(N)
LUT = [0] * len(N)

for t in range(0,len(N)):
    A = setupA(N[t],a1,a2,a3)
    b = np.zeros((N[t],1))
    for i in range(0,N[t]):
        b[i][0] = math.sin((i+1) * (f + 1))

    print("N = " + str(N[t]))
    print("Jacobi")
    start = time.time()
    jacobi(A,N[t])
    J[t] = time.time() - start
    print(J[t])
    print("Gauss - Seidl")
    
    start = time.time()
    gauss(A,N[t])
    G[t] = time.time() - start
    print(G[t])
    print("LU")
    start = time.time()
    LU(A,b,N[t])
    LUT[t] = time.time() - start
    print(LUT[t])
    # print(t)

fig1 = plt.gcf()
plt.plot(N,J)
plt.plot(N,G)
plt.plot(N,LUT)
plt.xlabel('Rozmiar macierzy A')
plt.ylabel('Czas [s]')
plt.title('Czas działania metod rozwiązania równań względem rozmiaru danych')
plt.legend(['Jacobi', 'Gauss-Seidl', 'LU'])
fig1.show()
fig1.savefig('foo.png')
#F