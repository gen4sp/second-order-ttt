"""Registered paired intervals, exact tied Page null and adverse completion bound."""
from __future__ import annotations

import math
from functools import lru_cache
from itertools import permutations, product

import numpy as np

PERMS = np.array(list(permutations(range(4))))
J = np.arange(1, 5)


def mcnemar(b, c):
    m = int(b + c)
    return 1.0 if m == 0 else sum(math.comb(m, j) for j in range(int(b), m + 1)) / 2**m


def cp(k, n):
    if not n:
        return None
    def tail(p, j):
        return sum(math.comb(n, i) * p**i * (1-p)**(n-i) for i in range(j, n+1))
    def solve(j, target):
        lo, hi = 0.0, 1.0
        for _ in range(64):
            mid = (lo + hi) / 2
            if tail(mid, j) < target:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2
    return [0.0 if k == 0 else solve(k, .025), 1.0 if k == n else solve(k+1, .975)]


def wilson(x, n, z=1.959963984540054):
    p = x/n
    c = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n))
    den = 1 + z*z/n
    return (p + z*z/(2*n)-c)/den, (p + z*z/(2*n)+c)/den


def newcombe10(a, b, c, d):
    n = a+b+c+d
    if not n:
        return None
    p1, p2 = (a+b)/n, (a+c)/n
    l1, u1 = wilson(a+b,n); l2, u2 = wilson(a+c,n)
    den = math.sqrt((a+b)*(c+d)*(a+c)*(b+d))
    phi = 0 if den == 0 else (a*d-b*c)/den
    dl = math.sqrt(max(0, (p1-l1)**2 - 2*phi*(p1-l1)*(u2-p2) + (u2-p2)**2))
    du = math.sqrt(max(0, (u1-p1)**2 - 2*phi*(u1-p1)*(p2-l2) + (p2-l2)**2))
    return [p1-p2-dl, p1-p2+du]


def paired(treated, control):
    t, c = np.asarray(treated, bool), np.asarray(control, bool)
    if t.shape != c.shape:
        raise ValueError('paired vectors must have identical shapes')
    a,b,cc,d = (int(np.sum(t&c)),int(np.sum(t&~c)),int(np.sum(~t&c)),int(np.sum(~t&~c)))
    n = len(t)
    return dict(n=n, table=dict(a=a,b=b,c=cc,d=d), treated=int(t.sum()), control=int(c.sum()),
                difference=(b-cc)/n if n else None, ci=newcombe10(a,b,cc,d),
                p=mcnemar(b,cc), treated_ci=cp(int(t.sum()),n), control_ci=cp(int(c.sum()),n))


def midranks_speed(tau):
    x = -np.asarray(tau, float)
    # Four columns only; broadcasting computes tie-aware mid-ranks exactly.
    return 1 + (x[:,None] > x[None,:]).sum(1) + ((x[:,None] == x[None,:]).sum(1)-1)/2


@lru_cache(maxsize=128)
def unit_null(r2):
    vals = np.asarray(r2, int)[PERMS] @ J
    lo = int(vals.min())
    return lo, np.bincount(vals-lo).astype(float)/24


@lru_cache(maxsize=8192)
def joint_null(patterns):
    lo, pmf = 0, np.ones(1)
    for p in patterns:
        offset, q = unit_null(p)
        lo += offset
        pmf = np.convolve(pmf, q)
    return lo, pmf


def page_from_ranks(ranks):
    r2 = np.rint(2*np.asarray(ranks)).astype(int).reshape(-1,4)
    patterns = tuple(sorted(tuple(sorted(r)) for r in r2))
    lo, pmf = joint_null(patterns)
    obs = int(np.sum(r2 @ J))
    return dict(L=obs/2, p=float(pmf[max(0,obs-lo):].sum()))


def page(tau):
    return page_from_ranks([midranks_speed(t) for t in tau])


def completion_options(tau, H=400):
    missing = [j for j,t in enumerate(tau) if t is None]
    if not missing:
        return [midranks_speed(tau)]
    observed = sorted({t for t in tau if t is not None})
    if any(t not in range(5,H+1,5) for t in observed):
        raise ValueError('outage outside registered completion domain')
    domain = np.arange(5,H+1,5)
    reps = set(observed)
    for a,b in zip([0]+observed,observed+[H+1]):
        reps.update(domain[(domain>a)&(domain<b)][:len(missing)].tolist())
    best = {}
    for fill in product(sorted(reps),repeat=len(missing)):
        t = list(tau)
        for j,v in zip(missing,fill):
            t[j]=v
        r = midranks_speed(t)
        key, contribution = tuple(sorted(r)), float(r@J)
        if key not in best or contribution < best[key][0]:
            best[key] = contribution,r
    return [v[1] for v in best.values()]


def worst_case_page(tau, H=400):
    # Called only with >=8 complete units of a frozen cohort <=12.
    options = [completion_options(t,H) for t in tau]
    count = math.prod(map(len,options))
    if count > 4096:
        return dict(p=1.0, combinations=count, reason='outside registered evaluable cohort')
    bound, arg = -1.0, None
    for combo in product(*options):
        p = page_from_ranks(combo)['p']
        if p > bound:
            bound,arg=p,[r.tolist() for r in combo]
    return dict(p=bound,combinations=count,completion=arg)


def dose_shape(rmst):
    x=np.asarray(rmst,float); delta=x[0]-x[-1]; adj=x[:-1]-x[1:]
    if abs(delta)<25: return 'FLAT'
    if delta<=-25: return 'REVERSED'
    if np.max(adj)>=.8*delta: return 'THRESHOLD'
    if np.min(adj)>=.1*delta: return 'GRADED'
    return 'IRREGULAR'


def bootstrap_rmst(tau, n_boot=10000):
    x = np.asarray(tau,float)
    if not len(x): return None
    rng=np.random.default_rng(0)
    means=x[rng.integers(0,len(x),(n_boot,len(x)))].mean(1)
    adj=means[:,:-1]-means[:,1:]
    return dict(rmst=x.mean(0).tolist(),ci=np.quantile(means,[.025,.975],axis=0).T.tolist(),
                adjacent=(x[:,:-1]-x[:,1:]).mean(0).tolist(),
                adjacent_ci=np.quantile(adj,[.025,.975],axis=0).T.tolist(),
                reverse_p=[mcnemar(int(np.sum(x[:,j]<x[:,j+1])),int(np.sum(x[:,j]>x[:,j+1])))
                           for j in range(x.shape[1]-1)])


def holm(pvalues, alpha):
    result=dict.fromkeys(pvalues,False)
    for i,(key,p) in enumerate(sorted(pvalues.items(),key=lambda kv:kv[1])):
        if p > alpha/(len(pvalues)-i): break
        result[key]=True
    return result
