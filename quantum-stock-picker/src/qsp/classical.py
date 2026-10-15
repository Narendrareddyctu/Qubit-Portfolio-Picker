"""The 'slow way': try every one of the 2^n portfolios."""
from __future__ import annotations
import time
import numpy as np
from .problem import PortfolioProblem, bits_of


def brute_force(prob: PortfolioProblem) -> dict:
    t0 = time.perf_counter()
    d = prob.cost_diagonal()
    n = prob.n
    idx = np.arange(2 ** n, dtype=np.int64)
    pop = np.zeros(2 ** n, dtype=np.int8)
    for i in range(n):
        pop += ((idx >> i) & 1).astype(np.int8)
    feasible = pop == prob.k
    best = int(np.argmin(np.where(feasible, d, np.inf)))
    return {
        "index": best, "x": bits_of(best, n), "cost": float(d[best]),
        "cost_min": float(d.min()), "cost_max": float(d.max()), "cost_mean": float(d.mean()),
        "n_feasible": int(feasible.sum()), "time_s": time.perf_counter() - t0,
        "popcount": pop, "feasible": feasible,
    }
