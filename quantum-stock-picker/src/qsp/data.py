"""Synthetic stock universe (seeded, reproducible, no internet needed).

Returns follow a simple sector-factor model so stocks in the same sector are
correlated. Swap `make_universe` for real data (e.g. yfinance) if you wish:
all the rest of the code only needs `mu` (expected return) and `cov`
(covariance matrix).
"""
from __future__ import annotations
import numpy as np

SECTORS = ["Tech", "Health", "Energy", "Finance", "Utilities"]
# Fictional tickers (stock i belongs to sector i % 5, so small universes stay diverse)
TICKERS = [
    "NVTX", "MEDA", "OILX", "BNKR", "PWRU", "CLDS", "GENO", "SOLR", "FINT", "HYDR",
    "CHIP", "VACC", "GASX", "PAYS", "GRID", "ROBO", "BIOL", "DRLL", "INSR", "WATR",
]


def make_universe(n: int = 20, seed: int = 7) -> dict:
    """Return dict with names, sectors, mu (annual return), cov (annual covariance)."""
    if n > len(TICKERS):
        raise ValueError(f"n must be <= {len(TICKERS)}")
    rng = np.random.default_rng(seed)
    n_all = len(TICKERS)
    sector = np.arange(n_all) % len(SECTORS)

    # Volatility per stock; higher vol tends to mean higher expected return (plus noise)
    vol = rng.uniform(0.12, 0.42, n_all)
    mu = 0.02 + 0.45 * vol + rng.normal(0, 0.035, n_all)

    # Factor-style correlations: market-wide part + extra for same sector
    market = rng.uniform(0.5, 1.0, n_all)
    sect_load = rng.uniform(0.4, 0.9, n_all)
    corr = np.zeros((n_all, n_all))
    for i in range(n_all):
        for j in range(n_all):
            if i == j:
                corr[i, j] = 1.0
            else:
                c = 0.25 * market[i] * market[j]
                if sector[i] == sector[j]:
                    c += 0.45 * sect_load[i] * sect_load[j]
                corr[i, j] = min(c, 0.95)
    cov = corr * np.outer(vol, vol)
    w, v = np.linalg.eigh(cov)                      # force positive semi-definite
    cov = (v * np.clip(w, 1e-6, None)) @ v.T

    idx = np.arange(n)                              # nested universes: 8 ⊂ 12 ⊂ 16 ⊂ 20
    return {
        "names": [TICKERS[i] for i in idx],
        "sectors": [SECTORS[sector[i]] for i in idx],
        "mu": mu[idx],
        "cov": cov[np.ix_(idx, idx)],
    }
