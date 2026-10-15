"""Portfolio selection as a QUBO / Ising problem.

Binary variable x_i = 1 if stock i is picked, 0 otherwise  (1 qubit per stock).

    minimise   C(x) = q * x^T Σ x  -  μ^T x  +  λ (Σ_i x_i - k)^2
                      └── risk ──┘   └profit┘   └─ pick exactly k stocks ─┘

Substituting x_i = (1 - z_i)/2 gives an Ising Hamiltonian
    H = const + Σ_i h_i Z_i + Σ_{i<j} J_ij Z_i Z_j
whose ground state encodes the best portfolio.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from qiskit.quantum_info import SparsePauliOp


@dataclass
class PortfolioProblem:
    mu: np.ndarray            # expected returns
    cov: np.ndarray           # covariance matrix
    k: int                    # number of stocks to pick
    q: float = 0.5            # risk aversion
    penalty: float | None = None
    normalise: bool = True    # rescale so the cost has unit std (keeps γ in a sane range)

    def __post_init__(self):
        self.n = len(self.mu)
        if self.penalty is None:
            # large enough that an infeasible bitstring never beats a feasible one
            self.penalty = 1.5 * (np.max(self.mu) + self.q * np.max(np.diag(self.cov)))
        self.scale = 1.0
        self._diag = None
        self._build_ising()
        if self.normalise:
            self.scale = 1.0 / self.cost_diagonal().std()
            self._diag = None
            self._build_ising()

    # ---------- QUBO -> Ising ----------
    def _build_ising(self):
        n, lam, k, s = self.n, self.penalty, self.k, self.scale
        A = self.q * self.cov + lam * np.ones((n, n))      # x^T A x
        b = -self.mu - 2 * lam * k * np.ones(n)            # b^T x
        c0 = lam * k * k
        off = A - np.diag(np.diag(A))
        h = -(np.diag(A) + b) / 2 - off.sum(axis=1) / 2
        J = np.triu(off / 2, 1)
        const = c0 + (np.diag(A) + b).sum() / 2 + off.sum() / 4
        self.h, self.J, self.const = h * s, J * s, const * s

    def qubo_cost(self, x) -> float:
        """Direct QUBO evaluation for one 0/1 vector (scaled like the Ising cost)."""
        x = np.asarray(x, float)
        return float(self.scale * (self.q * x @ self.cov @ x - self.mu @ x
                                   + self.penalty * (x.sum() - self.k) ** 2))

    def cost_diagonal(self) -> np.ndarray:
        """Ising energy of all 2^n bitstrings. Index bit i == stock i (qubit i)."""
        if self._diag is not None:
            return self._diag
        n = self.n
        idx = np.arange(2 ** n, dtype=np.int64)
        d = np.full(2 ** n, self.const)
        z = [(1 - 2 * ((idx >> i) & 1)).astype(np.float64) for i in range(n)]
        for i in range(n):
            d += self.h[i] * z[i]
            for j in range(i + 1, n):
                d += self.J[i, j] * z[i] * z[j]
        self._diag = d
        return d

    def hamiltonian(self) -> SparsePauliOp:
        """Qiskit operator (qubit i = stock i; constant dropped, it is a global phase)."""
        n, terms = self.n, []
        for i in range(n):
            lab = ["I"] * n; lab[n - 1 - i] = "Z"
            terms.append(("".join(lab), self.h[i]))
            for j in range(i + 1, n):
                lab = ["I"] * n; lab[n - 1 - i] = "Z"; lab[n - 1 - j] = "Z"
                terms.append(("".join(lab), self.J[i, j]))
        return SparsePauliOp.from_list(terms).simplify()

    def stats(self, x) -> dict:
        """Equal-weight portfolio profit (mean return) and risk (volatility)."""
        x = np.asarray(x, float)
        w = x / x.sum()
        return {"profit": float(self.mu @ w), "risk": float(np.sqrt(w @ self.cov @ w))}


def bits_of(index: int, n: int) -> np.ndarray:
    return np.array([(index >> i) & 1 for i in range(n)], dtype=int)
