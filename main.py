import sys, time
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
    t0 = time.time()

    n = 12
    k = 3

    parts_full = tuple(P(n, k))
    strats_full = tuple(S(parts_full))
    parts_full = sorted(set(tuple(sorted(s, reverse=True)) for s in strats_full))

    lpf = len(parts_full)
    lsf = len(strats_full)

    # Try subsets.
    combi = 0
    for r in range(lpf, 0, -1):
        for comb in combinations(range(lpf), r):
            combi += 1
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

# Sum of our probabilities should be 1.
            A_eq = sp.ones((1, l))
            b_eq = sp.ones(1)

# Each opponent strategy should yield zero or less. Note that if all of them
# yielded less than zero then we wouldn't be at an equilibrium (because an
# equilibrium in a symmetric zero-sum game has to have zero value for both
# players) but we don't need to worry about this because the opponent
# strategies are a superset of ours so one of them has to be the same and that
# one is guaranteed to have an expected value of zero.
            A_ub = MI
            b_ub = sp.zeros(lsf)

            t1 = time.time()
            print('Calling spo.linprog after %.3f seconds' % (t1-t0))
            res = spo.linprog(
                c = sp.ones(l),
                A_eq = A_eq,
                b_eq = b_eq,
                A_ub = A_ub,
                b_ub = b_ub,
                bounds = (0., 1.),
            )
            if res['success']:
                pprint(('n, k:', (n,k)))
                pprint(('combi:', combi))
                pprint(('len(comb):', len(comb)))
                pprint(('MI shape:', MI.shape))
                pprint(('MI rank:', nl.matrix_rank(MI)))
                pprint(sorted(set(
                    row for row in zip(
                        [tuple(sorted(s, reverse=True)) for s in strats],
                        ['%.5f' % n for n in res['x']],
                        -A_ub.sum(0),
                    ) if float(row[1]) > 1e-5
                ), key=lambda row: row[1], reverse=True))
                pprint(sum(res['x']))
                sys.exit()


if __name__ == '__main__':
    main()
