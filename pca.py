import math
import random


def vadd(a, b):
    return [a[i] + b[i] for i in range(len(a))]


def vsubt(a, b):
    return [a[i] - b[i] for i in range(len(a))]


def madd(A, B):
    return [vadd(A[i], B[i]) for i in range(len(A))]


def msubt(A, B):
    return [vsubt(A[i], B[i]) for i in range(len(A))]


def sadd(a, scalar):
    return [a[i] + scalar for i in range(len(a))]


def smadd(A, scalar):
    return [sadd(A[i], scalar) for i in range(len(A))]


def ssubt(a, scalar):
    return [a[i] - scalar for i in range(len(a))]


def smsubt(A, scalar):
    return [ssubt(A[i], scalar) for i in range(len(A))]


def smult(a, scalar):
    return [a[i] * scalar for i in range(len(a))]


def smmult(A, scalar):
    return [smult(A[i], scalar) for i in range(len(A))]


def sdiv(a, scalar):
    return smult(a, 1 / scalar)


def smdiv(A, scalar):
    return [sdiv(A[i], scalar) for i in range(len(A))]


def dot(a, b):
    return sum([a[i] * b[i] for i in range(len(a))])


def mdot(A, b):
    return [dot(A[i], b) for i in range(len(A))]


def outer(a, b):
    return [[a[i] * b[j] for i in range(len(a))] for j in range(len(b))]


def transpose(A):
    return list(map(list, zip(*A)))


def norm(a):
    return math.sqrt(sum([pow(a[i], 2) for i in range(len(a))]))


def normalize(A):
    return [sdiv(a, norm(a)) for a in A]


def distance(a, b):
    return norm(vsubt(a, b))


def mean(A):
    return sdiv([sum([a[i] for a in A]) for i in range(len(A[0]))], len(A))


def center(A, b):
    return [vsubt(a, b) for a in A]


def covariance(A):
    outer_products = [outer(a, a) for a in A]
    return smdiv([[sum([o[i][j] for o in outer_products]) for i in range(len(A[0]))] for j in range(len(A[0]))], len(A))


def rayleigh_quotient(A, b):
    return dot(b, mdot(A, b)) / dot(b, b)


def hotelling_deflation(A, vector, value):
    return msubt(A, smmult(outer(vector, vector), value))


def power_iteration(A):
    bk = [random.random() for i in range(len(A))]
    bk = sdiv(bk, norm(bk))
    rk = rayleigh_quotient(A, bk)

    while True:
        bk1 = mdot(A, bk)
        bk1 = sdiv(bk1, norm(bk1))
        rk1 = rayleigh_quotient(A, bk1)

        if math.isclose(rk1, rk):
            return bk1, rk1

        bk = bk1
        rk = rk1


def spectral_decomposition(A, n):
    vectors = [[0.0 for i in range(n)] for j in range(n)]
    values = [0.0 for i in range(n)]

    B = A
    for i in range(n):
        vectors[i], values[i] = power_iteration(B)
        B = hotelling_deflation(B, vectors[i], values[i])

    return vectors, values


def pca(A, n):
    no_transpose = len(A) >= len(A[0])  # TODO: we seem to be loosing a little bit of accuracy when we do the transpose trick
    B = normalize(center(A, mean(A))) if no_transpose else transpose(normalize(center(A, mean(A))))
    C = covariance(B)
    eigen_vectors, eigen_values = spectral_decomposition(C, n)
    components = eigen_vectors if no_transpose else [mdot(B, eigen_vector) for eigen_vector in eigen_vectors]
    return components
