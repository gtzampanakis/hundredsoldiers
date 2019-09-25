from itertools import permutations
from pmemoize import MemoizedFunction

@MemoizedFunction
def PC(n, k, x=-1):
    """
    Partitions of n into k, not allowing zeros, with the maximum number allowed
    being x
    """
    if k > n:
        return
    if k == 1:
        yield (n,)
    else:
        q = n//k
        r = n%k
        a = q+1 if r else q
        for m in range(a, n-k+2):
            if x != -1 and m > x:
                break
            for pj in PC(n-m, k-1, m):
                yield (m,) + pj

def P(n, k):
    """
    Partitions of n into k, not allowing zeros.
    """
    if k > n:
        return
    res = []
    for l in range(1, k+1):
        for pc in PC(n, l):
            yield pc + (0,) * (k-l)

def S(n, k):
    """
    Available strategies.
    """
    for pj in P(n, k):
        for c in permutations(pj):
            yield c

from pprint import pprint
pprint(list(S(4, 2)))
