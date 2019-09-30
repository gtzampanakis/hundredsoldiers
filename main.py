import sys
from pprint import pprint
from itertools import combinations, permutations

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
def S(parts):
    """
    Available strategies.
    """
    res = []
    for pj in parts:
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

def main():
    n = 5
    k = 3

    parts_full = tuple(P(n, k))

    # Remove dominated strategies.
    strats_full = tuple(S(parts_full))
    while 1:
        l = len(strats_full)

        MI = sp.zeros((l, l))
        for i in range(l):
            for j in range(l):
                MI[i,j] = M(strats_full, i, j)

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

        print('Removing %s dominated strategies: %s' % (
            len(dominated), [strats_full[d] for d in dominated]
        ))

        strats_full = tuple(s for (si, s) in enumerate(strats_full) if si not in dominated)

    parts_full = sorted(set(tuple(sorted(s, reverse=True)) for s in strats_full))
    lpf = len(parts_full)
    lsf = len(strats_full)

    # Try subsets.
    for r in range(lpf, 0, -1):
        for comb in combinations(range(lpf), r):
            parts = tuple(parts_full[i] for i in comb)
            strats = tuple(S(parts))
            l = len(strats)

            MI = sp.zeros((lsf, l))
            for i in range(lsf):
                for j in range(l):
                    MI[i,j] = match(
                        strats_full[i],
                        strats[j],
                    )

            A_eq = sp.vstack((
                MI,
                sp.ones((1, l)),
            ))

            b_eq = sp.concatenate((
                sp.zeros(lsf),
                sp.ones(1),
            ))

            res = spo.linprog(
                c = sp.zeros(l),
                A_eq = A_eq,
                b_eq = b_eq,
                bounds = (0., 1.),
            )
            print(res)
            if res['success']:
                pprint(res)
                pprint('')
                pprint(('comb:', comb))
                pprint(('MI shape:', MI.shape))
                pprint(('MI rank:', nl.matrix_rank(MI)))
                pprint(list(zip(
                    strats,
                    ['%.5f' % n for n in res['x']],
                    MI.sum(1),
                )))
                pprint(sum(res['x']))
                sys.exit()


if __name__ == '__main__':
    main()
