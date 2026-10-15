# Quantum Stock Picker

Pick a few stocks out of many for **high profit and low risk**, using the quantum algorithm **QAOA** built in **Qiskit**, then check how close it gets to the true best answer.

![frontend](docs/frontend_screenshot.png)

## The idea in five steps

| Step | What we do | Where |
|---|---|---|
| 1. Set up | Each stock is picked (1) or not (0): **one qubit per stock**. Cost = `q·xᵀΣx − μᵀx + λ(Σx − k)²` (risk − profit + "pick exactly k" penalty), converted to an Ising Hamiltonian. | `src/qsp/problem.py` |
| 2. Solve | QAOA (depth p = 3) in Qiskit; COBYLA tunes the angles; 2048 measurements; keep the cheapest valid basket. | `src/qsp/qaoa.py` |
| 3. Check | Compare with brute force over all 2ⁿ baskets. | `src/qsp/classical.py` |
| 4. Scale | Repeat for 8, 12, 16 and 20 stocks. | `run_experiments.py` |
| 5. Show | Risk-vs-profit charts with the quantum picks marked: a web page, PNG figures and a PDF deck. | `web/`, `make_figures.py`, `docs/` |

## Results (seed 7, p = 3, noise-free simulation)

| Stocks (qubits) | Pick k | Baskets | Found the best? | Rank of QAOA pick | Approx. ratio (random) | Chance one shot is the best (random) |
|---|---|---|---|---|---|---|
| 8  | 3 | 256       | **yes** | 1 of 56     | 0.976 (0.873) | 1.3% (0.39%) |
| 12 | 4 | 4,096     | **yes** | 1 of 495    | 0.986 (0.888) | 0.12% (0.024%) |
| 16 | 5 | 65,536    | no | 3 of 4,368  | 0.991 (0.890) | 1 in 8,300 (1 in 66,000) |
| 20 | 6 | 1,048,576 | no | 8 of 38,760 | 0.992 (0.891) | 1 in 92,000 (1 in 1,050,000) |

*Approximation ratio = (worst cost − QAOA average cost) / (worst − best); 1 is perfect.*

What to take from it:

* At 8 and 12 stocks QAOA returns exactly the best portfolio found by brute force.
* At 16 and 20 stocks the pick is still among the best few valid portfolios (3 of 4,368; 8 of 38,760), sharing 4 of 5 and 5 of 6 stocks with the best one.
* The chance that a *single* measurement is the best portfolio drops with size, while staying better than random guessing at every size (about 3x at 8 stocks, rising to about 11x at 20). Keeping the best of 2048 shots is what keeps the answer good.
* Be honest about scope: this is a **classical simulation** of QAOA on **synthetic data**. Brute force is faster than the QAOA simulation at these sizes, so this project demonstrates the method and how it scales; it does not demonstrate quantum advantage. Real hardware would add noise.

## Project layout

```
quantum-stock-picker/
├── README.md  requirements.txt  LICENSE  pytest.ini  .gitignore
├── run_experiments.py        # QAOA vs brute force for 8/12/16/20 stocks -> results/
├── make_figures.py           # PNG charts -> results/figures/
├── make_presentation.py      # PDF slides -> docs/
├── src/qsp/                  # data.py, problem.py, qaoa.py, classical.py
├── tests/test_problem.py     # Ising == QUBO, circuit == Qiskit QAOAAnsatz, etc.
├── results/                  # results.json, run_log.txt, figures/
├── web/                      # index.html, style.css, app.js, results.js
└── docs/                     # Quantum_Stock_Picker_Presentation.pdf, screenshot
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                       # 5 tests
python run_experiments.py       # ~4 minutes (the 20-qubit run is the slow part)
python run_experiments.py --sizes 8 12      # quick version
python make_figures.py
python make_presentation.py
```

Open `web/index.html` in a browser (no server needed; it reads `web/results.js`). To publish it, enable **GitHub Pages** and serve the `web/` folder (or copy it to `docs/`).

## How the code works

* **Data** (`data.py`): 20 fictional stocks in 5 sectors with a seeded factor model, so same-sector stocks are correlated. Smaller universes are the first n stocks. To use real data, replace `make_universe` with anything that returns `mu` (expected returns) and `cov` (covariance).
* **Problem** (`problem.py`): builds the Ising coefficients and the Qiskit `SparsePauliOp`. Tests confirm the Ising energy of every bitstring equals the QUBO cost.
* **QAOA** (`qaoa.py`): `H` on every qubit, then p layers of `RZ`/`RZZ` (cost) and `RX` (mixer). Up to 12 qubits the circuit runs on Qiskit's `Statevector`. Above that a NumPy statevector engine does the identical maths because it is much faster; a test checks both engines agree and that the circuit matches Qiskit's `QAOAAnsatz`.
* **Read-out**: sample 2048 bitstrings from the final state, keep the cheapest one with exactly k stocks. A random-guess baseline with the same number of shots is recorded for comparison.
* **Profit and risk** of a basket use equal weights: profit = mean expected return, risk = √(wᵀΣw).

## Ideas to extend

Real market data (yfinance), constraint-preserving mixers (XY / Dicke warm start) so no shots are wasted on invalid baskets, deeper p, noisy simulators or real IBM hardware via `qiskit-ibm-runtime`, weights instead of in/out choices.

## Disclaimer

Educational project with synthetic data. Not investment advice.
