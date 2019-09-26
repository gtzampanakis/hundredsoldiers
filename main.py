from pprint import pprint
from itertools import permutations

import numpy as np
import scipy as sp
import scipy.linalg as spl
import scipy.optimize as spo
import numpy.linalg as nl

from pmemoize import MemoizedFunction

@MemoizedFunction
def PC(n, k, x=-1):
    """
    Partitions of n into k, not allowing zeros, with the maximum number allowed
    being x
    """
    res = []
    if k > n:
        pass
    elif k == 1:
        res.append((n,))
    else:
        q = n//k
        r = n%k
        a = q+1 if r else q
        for m in range(a, n-k+2):
            if x != -1 and m > x:
                break
            for pj in PC(n-m, k-1, m):
                res.append((m,) + pj)
    return res

@MemoizedFunction
def P(n, k):
    """
    Partitions of n into k, not allowing zeros.
    """
    res = []
    if k > n:
        pass
    else:
        for l in range(1, k+1):
            for pc in PC(n, l):
                res.append(pc + (0,) * (k-l))
    return res

@MemoizedFunction
def S(n, k):
    """
    Available strategies.
    """
    res = []
    for pj in P(n, k):
        for c in permutations(pj):
            if c not in res:
                res.append(c)
    return res

@MemoizedFunction
def match(s1, s2):
    w, l = 0, 0
    for a, b in zip(s1, s2):
        if a > b:
            w += 1
        elif a < b:
            l += 1
    if w > l:
        return 1
    elif w < l:
        return -1
    else:
        return 0

@MemoizedFunction
def M(strats, i, j):
    s1 = strats[i]
    s2 = strats[j]
    return match(s1, s2)

def exps(a, n, k):
    lstrats = len(S(n, k))
    res = [0] * lstrats
    for i in range(lstrats):
        for j in range(lstrats):
            res[i] += M(i, j, n, k) * a[j]
    return res

def main():
    n = 4
    k = 3

    strats = tuple(S(n, k))

    # Remove dominated strategies.
    while 1:
        l = len(strats)

        MI = sp.zeros((l, l))
        for i in range(l):
            for j in range(l):
                MI[i,j] = M(strats, i, j)

        dominated = []
        for i1 in range(l):
            for i2 in range(i1 + 1, l):
                if i1 in dominated or i2 in dominated:
                    continue
                diff = MI[i1,:] >= MI[i2,:]
                if all(diff):
                    dominated.append(i2)
                diff = MI[i1,:] <= MI[i2,:]
                if all(diff):
                    dominated.append(i1)

        if not dominated:
            break

        strats = tuple(s for (si, s) in enumerate(strats) if si not in dominated)

        print('Removed %s dominated strategies' % len(dominated))
        
    A = sp.vstack((MI, sp.ones((1, l))))
    B = sp.array(([0.] * l) + [1.])

    def fun(x):
        return sp.dot(A, x) - B

    res = spo.least_squares(
        fun = fun,
        x0 = sp.ones(l) / l,
        bounds = (0., 1.),
    )
    pprint(res)
    pprint('')
    pprint(('MI shape:', MI.shape))
    pprint(('MI rank:', nl.matrix_rank(MI)))
    pprint(list(zip(
        strats,
        ['%.5f' % n for n in res['x']],
        MI.sum(1),
    )))
    pprint(sum(res['x']))


if __name__ == '__main__':
    main()
