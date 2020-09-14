import math
import random


def vadd(a, b):
    n = len(a)
    return [a[i] + b[i] for i in range(n)]


def vsubt(a, b):
    n = len(a)
    return [a[i] - b[i] for i in range(n)]


def madd(A, B):
    n = len(A)
    return [vadd(A[i], B[i]) for i in range(n)]


def msubt(A, B):
    n = len(A)
    return [vsubt(A[i], B[i]) for i in range(n)]


def sadd(a, scalar):
    n = len(a)
    return [a[i] + scalar for i in range(n)]


def smadd(A, scalar):
    n = len(A)
    return [sadd(A[i], scalar) for i in range(n)]


def ssubt(a, scalar):
    n = len(a)
    return [a[i] - scalar for i in range(n)]


def smsubt(A, scalar):
    n = len(A)
    return [ssubt(A[i], scalar) for i in range(n)]


def smult(a, scalar):
    n = len(a)
    return [a[i] * scalar for i in range(n)]


def smmult(A, scalar):
    n = len(A)
    return [smult(A[i], scalar) for i in range(n)]


def sdiv(a, scalar):
    n = len(a)
    reciprocal = 1 / scalar
    return smult(a, reciprocal)


def smdiv(A, scalar):
    n = len(A)
    reciprocal = 1 / scalar
    return [sdiv(A[i], reciprocal) for i in range(n)]


def dot(a, b):
    n = len(a)
    return sum([a[i] * b[i] for i in range(n)])


def mdot(A, b):
    n = len(A)
    return [dot(A[i], b) for i in range(n)]


def outer(a, b):
    A = [[0.0 for i in range(len(a))] for j in range(len(b))]
    for i in range(len(a)):
        for j in range(len(b)):
            A[i][j] = A[i][j] + a[i] * b[j]
    return A


def norm(vector):
    n = len(vector)
    return math.sqrt(sum([pow(vector[i], 2) for i in range(n)]))


def mean(data):
    data_size = len(data)
    datum_size = len(data[0])
    M = [0.0 for i in range(datum_size)]

    for datum in data:
        M = vadd(M, datum)
    M = sdiv(M, data_size)

    return M


def center(data, P):
    n = len(data)
    return [vsubt(data[i], P) for i in range(n)]


def covariance(data):
    data_size = len(data)
    datum_size = len(data[0])
    C = [[0.0 for i in range(datum_size)] for j in range(datum_size)]

    for datum in data:
        C = madd(C, outer(datum, datum))
    C = smdiv(C, data_size - 1)

    return C


def power_iteration(matrix):
    bk = [random.random() for i in range(len(matrix))]
    bk = sdiv(bk, norm(bk))
    rk = rayleigh_quotient(matrix, bk)

    while True:
        bk1 = mdot(matrix, bk)
        bk1 = sdiv(bk1, norm(bk1))
        rk1 = rayleigh_quotient(matrix, bk1)

        if math.isclose(rk1, rk):
            return bk1, rk1

        bk = bk1
        rk = rk1


def rayleigh_quotient(matrix, vector):
    R = dot(vector, mdot(matrix, vector)) / dot(vector, vector)
    return R


def hotelling_deflation(matrix, vector, value):
    B = msubt(matrix, smmult(outer(vector, vector), value))
    return B


def spectral_decomposition(matrix, n):
    vectors = [[0.0 for i in range(n)] for j in range(n)]
    values = [0.0 for i in range(n)]

    B = matrix
    for i in range(n):
        vectors[i], values[i] = power_iteration(B)
        B = hotelling_deflation(B, vectors[i], values[i])

    return vectors, values


def pca(data, num_components):
    data_mean = mean(data)
    data = center(data, data_mean)
    covariance_matrix = covariance(data)
    eigen_vectors, eigen_values = spectral_decomposition(covariance_matrix, num_components)
    return eigen_vectors


training_data = [
    [7, 4, 3],
    [4, 1, 8],
    [6, 3, 5],
    [8, 6, 1],
    [8, 5, 7],
    [7, 2, 9],
    [5, 3, 3],
    [9, 5, 8],
    [7, 4, 5],
    [8, 2, 2]
]
components = pca(training_data, 3)
print(components)
