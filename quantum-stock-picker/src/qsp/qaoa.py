"""QAOA for the portfolio Hamiltonian.

Two interchangeable simulation engines run the *same* QAOA circuit
    |ψ(β,γ)> = Π_l  e^{-iβ_l ΣX}  e^{-iγ_l H}  |+>^n
* "qiskit": explicit Qiskit circuit (RZ/RZZ/RX gates), simulated with Statevector
* "fast"  : hand-written NumPy statevector simulator (same maths, much faster)
`engine="auto"` uses Qiskit up to 12 qubits and the fast engine above that.
The test-suite checks that both engines give identical probabilities.
"""
from __future__ import annotations
import time
import numpy as np
from scipy.optimize import minimize
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
from .problem import PortfolioProblem, bits_of


def qaoa_circuit(prob: PortfolioProblem, p: int):
    """Explicit QAOA circuit: H^n, then p x [cost layer (RZ, RZZ) + mixer layer (RX)]."""
    n = prob.n
    gam, bet = ParameterVector("γ", p), ParameterVector("β", p)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for l in range(p):
        for i in range(n):
            qc.rz(2 * gam[l] * prob.h[i], i)                  # e^{-iγ h_i Z_i}
            for j in range(i + 1, n):
                qc.rzz(2 * gam[l] * prob.J[i, j], i, j)       # e^{-iγ J_ij Z_i Z_j}
        qc.rx(2 * bet[l], range(n))                           # e^{-iβ X_i}
    return qc, gam, bet


class QiskitEngine:
    name = "qiskit"

    def __init__(self, prob: PortfolioProblem, p: int):
        self.n, self.p = prob.n, p
        self.qc, self.gam, self.bet = qaoa_circuit(prob, p)

    def circuit(self, params):
        return self.qc.assign_parameters({**dict(zip(self.gam, params[: self.p])),
                                          **dict(zip(self.bet, params[self.p:]))})

    def probabilities(self, params):
        return Statevector(self.circuit(params)).probabilities()


class FastEngine:
    name = "fast"

    def __init__(self, prob: PortfolioProblem, p: int):
        self.n, self.p = prob.n, p
        self.diag = prob.cost_diagonal()

    def probabilities(self, params):
        n, p = self.n, self.p
        gam, bet = params[:p], params[p:]
        psi = np.full(2 ** n, 2 ** (-n / 2), dtype=np.complex128)
        for l in range(p):
            psi *= np.exp(-1j * gam[l] * self.diag)          # cost layer
            c, s = np.cos(bet[l]), -1j * np.sin(bet[l])
            for q in range(n):                                # mixer: RX(2β) on every qubit
                v = psi.reshape(-1, 2, 2 ** q)
                a, b = v[:, 0, :].copy(), v[:, 1, :].copy()
                v[:, 0, :] = c * a + s * b
                v[:, 1, :] = s * a + c * b
        return np.abs(psi) ** 2


def make_engine(prob, p, engine="auto"):
    if engine == "auto":
        engine = "qiskit" if prob.n <= 12 else "fast"
    return QiskitEngine(prob, p) if engine == "qiskit" else FastEngine(prob, p)


def run_qaoa(prob: PortfolioProblem, bf: dict, p: int = 3, restarts: int = 3,
             maxiter: int = 150, shots: int = 2048, seed: int = 0, engine: str = "auto") -> dict:
    """Optimise (γ,β) with COBYLA, then 'measure' the circuit to read out a portfolio."""
    rng = np.random.default_rng(seed)
    eng = make_engine(prob, p, engine)
    diag, n = prob.cost_diagonal(), prob.n
    history: list[float] = []

    def energy(params):
        e = float(eng.probabilities(params) @ diag)
        history.append(e)
        return e

    t0 = time.perf_counter()
    best_res, best_hist = None, None
    for r in range(restarts):
        history.clear()
        if r == 0:      # linear-ramp start (annealing-inspired)
            x0 = np.concatenate([np.linspace(0.1, 0.8, p), np.linspace(0.8, 0.1, p)])
        else:
            x0 = np.concatenate([rng.uniform(0, 1.0, p), rng.uniform(0, np.pi / 2, p)])
        res = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter, "rhobeg": 0.3})
        if best_res is None or res.fun < best_res.fun:
            best_res, best_hist = res, list(history)
    runtime = time.perf_counter() - t0

    probs = eng.probabilities(best_res.x)
    probs = probs / probs.sum()
    exp_cost = float(probs @ diag)
    cmin, cmax = bf["cost_min"], bf["cost_max"]

    # read-out: sample bitstrings like a real device, keep the best feasible one
    samples = rng.choice(len(probs), size=shots, p=probs)
    feas = bf["popcount"][samples] == prob.k
    s_cost = diag[samples]
    best = int(samples[np.argmin(np.where(feas, s_cost, np.inf))]) if feas.any() else int(samples[np.argmin(s_cost)])
    # baseline: same number of purely random guesses
    rnd = rng.integers(0, 2 ** n, size=shots)
    rfeas = bf["popcount"][rnd] == prob.k
    rbest = int(rnd[np.argmin(np.where(rfeas, diag[rnd], np.inf))]) if rfeas.any() else None

    opt = bf["index"]
    rank = int((diag[bf["feasible"]] < diag[best]).sum()) + 1     # 1 = optimal
    return {
        "engine": eng.name, "p": p, "n_evals": int(best_res.nfev), "runtime_s": runtime,
        "params": best_res.x.tolist(), "history": best_hist,
        "expected_cost": exp_cost,
        "approx_ratio": (cmax - exp_cost) / (cmax - cmin),
        "random_approx_ratio": (cmax - bf["cost_mean"]) / (cmax - cmin),
        "prob_optimal": float(probs[opt]),
        "prob_feasible": float(probs[bf["feasible"]].sum()),
        "best_index": best, "x": bits_of(best, n), "best_cost": float(diag[best]),
        "best_feasible": bool(bf["popcount"][best] == prob.k),
        "rank_among_feasible": rank, "is_optimal": best == opt,
        "random_best_index": rbest, "probs": probs,
    }
