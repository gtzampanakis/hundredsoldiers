from pprint import pprint
from itertools import permutations
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
def M(i, j, n, k):
    strats = S(n, k)
    s1 = strats[i]
    s2 = strats[j]
    return match(s1, s2)

def loss(a, n, k):
    res = [0] * k
    for i in range(k):
        acc = 0.
        for j in range(k):
            res[i] += M(i, j, n, k) * a[j]
    return sum(res)

N = 10
K = 4

a0 = [9, 2, 2, 1]
a0 = [ai/float(K) for ai in a0]

print(loss(a0, N, K))
