import numpy as np
from qsp import make_universe, PortfolioProblem, brute_force
from qsp.qaoa import QiskitEngine, FastEngine
from qsp.problem import bits_of


def _prob(n=6, k=3, normalise=False):
    u = make_universe(n)
    return PortfolioProblem(u["mu"], u["cov"], k, normalise=normalise)


def test_ising_matches_qubo():
    p = _prob()
    d = p.cost_diagonal()
    for i in range(2 ** p.n):
        assert abs(d[i] - p.qubo_cost(bits_of(i, p.n))) < 1e-9


def test_hamiltonian_matches_diagonal():
    p = _prob(5, 2)
    H = p.hamiltonian().to_matrix().diagonal().real
    d = p.cost_diagonal()
    assert np.allclose(H - H[0], d - d[0])


def test_brute_force_is_feasible_and_optimal():
    p = _prob(8, 3, True)
    bf = brute_force(p)
    assert bf["x"].sum() == p.k
    d = p.cost_diagonal()
    assert bf["cost"] == d[bf["feasible"]].min()


def test_engines_agree():
    p = _prob(6, 3, True)
    params = np.array([0.3, 0.5, 0.7, 0.6, 0.4, 0.2])
    a = QiskitEngine(p, 3).probabilities(params)
    b = FastEngine(p, 3).probabilities(params)
    assert np.allclose(a, b, atol=1e-9)


def test_circuit_matches_qiskit_qaoa_ansatz():
    """Our explicit circuit must equal Qiskit's own QAOAAnsatz (up to global phase)."""
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit.quantum_info import Statevector
    p = _prob(5, 2, True)
    ans = QAOAAnsatz(cost_operator=p.hamiltonian(), reps=2)
    prm = np.array([0.4, 0.7, 0.5, 0.3])           # gamma0, gamma1, beta0, beta1
    names = {q.name: q for q in ans.parameters}
    bound = ans.assign_parameters({names["γ[0]"]: prm[0], names["γ[1]"]: prm[1],
                                   names["β[0]"]: prm[2], names["β[1]"]: prm[3]})
    ref = Statevector(bound).probabilities()
    mine = QiskitEngine(p, 2).probabilities(prm)
    assert np.allclose(ref, mine, atol=1e-9)
