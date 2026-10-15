"""Create PNG charts from results/results.json (used by README and the PDF)."""
import json, sys, warnings
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
OUT = ROOT / "results" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
R = json.loads((ROOT / "results" / "results.json").read_text())
QC, CC, DIM, INK = "#e0870b", "#0f7c8a", "#9aa9b4", "#15222c"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK})
pc = lambda v, _=None: f"{100 * v:.0f}%"


def axes_fmt(ax):
    ax.xaxis.set_major_formatter(pc); ax.yaxis.set_major_formatter(pc)
    ax.set_xlabel("Risk (volatility per year)"); ax.set_ylabel("Profit (expected return per year)")
    ax.grid(alpha=.25)


for e in R["sizes"]:
    n = e["n"]
    q, b = set(e["qaoa"]["picked"]), set(e["brute"]["picked"])
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for s in e["stocks"]:
        isq, isb = s["name"] in q, s["name"] in b
        if isb:
            ax.scatter(s["risk"], s["profit"], s=260, facecolors="none", edgecolors=CC, linewidths=2.2, zorder=2)
        ax.scatter(s["risk"], s["profit"], s=90 if isq else 40, color=QC if isq else DIM, zorder=3,
                   alpha=1 if isq else .75)
        if isq or isb:
            ax.annotate(s["name"], (s["risk"], s["profit"]), xytext=(9, 7), textcoords="offset points", fontweight="bold")
    ax.margins(.14)
    axes_fmt(ax)
    ax.set_title(f"{n} stocks: risk vs profit (orange = QAOA pick, teal ring = best possible)", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f"stocks_n{n}.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    ax.scatter([c["risk"] for c in e["cloud"]], [c["profit"] for c in e["cloud"]], s=10, color=DIM, alpha=.45,
               label=f"random valid {e['k']}-stock portfolios")
    ax.scatter(e["brute"]["risk"], e["brute"]["profit"], marker="D", s=170, facecolors="none", edgecolors=CC,
               linewidths=2.5, label="best possible (exhaustive)", zorder=3)
    ax.scatter(e["qaoa"]["risk"], e["qaoa"]["profit"], s=60, color=QC, label="QAOA pick", zorder=4)
    axes_fmt(ax); ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"{n} stocks: whole portfolios", fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / f"portfolios_n{n}.png", dpi=170); plt.close(fig)

ns = [e["n"] for e in R["sizes"]]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
a1.plot(ns, [e["qaoa"]["approx_ratio"] for e in R["sizes"]], "o-", color=QC, lw=2.5, label="QAOA")
a1.plot(ns, [e["qaoa"]["random_approx_ratio"] for e in R["sizes"]], "o--", color=DIM, lw=2, label="random guessing")
a1.set_xticks(ns); a1.set_xlabel("Number of stocks (qubits)"); a1.set_ylabel("Approximation ratio (1 = perfect)")
a1.set_ylim(.8, 1.005); a1.legend(frameon=False); a1.grid(alpha=.25)
a2.semilogy(ns, [e["qaoa"]["prob_optimal"] for e in R["sizes"]], "o-", color=QC, lw=2.5, label="QAOA")
a2.semilogy(ns, [e["qaoa"]["random_prob_optimal"] for e in R["sizes"]], "o--", color=DIM, lw=2, label="random guessing")
a2.set_xticks(ns); a2.set_xlabel("Number of stocks (qubits)"); a2.set_ylabel("Chance one shot is the best portfolio")
a2.legend(frameon=False); a2.grid(alpha=.25, which="both")
fig.tight_layout(); fig.savefig(OUT / "scaling.png", dpi=170); plt.close(fig)

fig, ax = plt.subplots(figsize=(7.2, 4))
for e in R["sizes"]:
    h = e["qaoa"]["history"]; ax.plot(range(len(h)), h, lw=1.6, label=f"{e['n']} stocks")
ax.set_xlabel("Optimiser evaluation"); ax.set_ylabel("Average cost (lower is better)")
ax.legend(frameon=False); ax.grid(alpha=.25); fig.tight_layout()
fig.savefig(OUT / "convergence.png", dpi=170); plt.close(fig)

# circuit drawing (4 stocks, p=1) for the README / slides
sys.path.insert(0, str(ROOT / "src"))
from qsp import make_universe, PortfolioProblem
from qsp.qaoa import qaoa_circuit
u = make_universe(3); pb = PortfolioProblem(u["mu"], u["cov"], 1)
qc, _, _ = qaoa_circuit(pb, 1)
qc.draw("mpl", filename=str(OUT / "circuit_n4.png"), fold=-1, scale=1.0)
print("figures saved to", OUT)
