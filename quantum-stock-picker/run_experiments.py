"""Run QAOA vs brute force for 8/12/16/20 stocks and save results (JSON + JS for the website).

    python run_experiments.py                 # full run (8, 12, 16, 20)
    python run_experiments.py --sizes 8 12    # quicker
"""
from __future__ import annotations
import argparse, json, sys, warnings
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
warnings.filterwarnings("ignore")
from qsp import make_universe, PortfolioProblem, brute_force, run_qaoa  # noqa: E402

ROOT = Path(__file__).parent
CFG = {   # n: (k, restarts, maxiter)
    8: (3, 4, 200), 12: (4, 3, 200), 16: (5, 2, 150), 20: (6, 2, 120),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[8, 12, 16, 20])
    ap.add_argument("--layers", type=int, default=3, help="QAOA depth p")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    uni_full = make_universe(20, args.seed)
    rng = np.random.default_rng(1)
    out = {"seed": args.seed, "layers": args.layers, "sizes": [],
           "universe": [{"name": n, "sector": s, "profit": float(m), "risk": float(np.sqrt(c))}
                        for n, s, m, c in zip(uni_full["names"], uni_full["sectors"],
                                              uni_full["mu"], np.diag(uni_full["cov"]))]}
    for n in args.sizes:
        k, restarts, maxiter = CFG.get(n, (n // 4 + 1, 2, 150))
        u = make_universe(n, args.seed)
        prob = PortfolioProblem(u["mu"], u["cov"], k, q=0.5)
        bf = brute_force(prob)
        res = run_qaoa(prob, bf, p=args.layers, restarts=restarts, maxiter=maxiter, seed=n)
        d = prob.cost_diagonal()

        # cloud of random feasible portfolios for the risk/profit chart
        feas_idx = np.flatnonzero(bf["feasible"])
        pick = rng.choice(feas_idx, size=min(500, len(feas_idx)), replace=False)
        from qsp.problem import bits_of
        cloud = [{**prob.stats(bits_of(int(i), n)), "cost": float(d[i])} for i in pick]
        stat = lambda x: prob.stats(x)
        sel = lambda x: [u["names"][i] for i in range(n) if x[i]]
        # top-probability feasible portfolios found by QAOA
        pr = res["probs"] * bf["feasible"]
        top = np.argsort(pr)[::-1][:5]
        rb = res["random_best_index"]
        entry = {
            "n": n, "k": k, "n_states": 2 ** n, "n_feasible": bf["n_feasible"],
            "names": u["names"],
            "stocks": [{"name": u["names"][i], "sector": u["sectors"][i], "profit": float(u["mu"][i]),
                        "risk": float(np.sqrt(u["cov"][i, i]))} for i in range(n)],
            "brute": {"picked": sel(bf["x"]), "cost": bf["cost"], "time_s": bf["time_s"], **stat(bf["x"])},
            "qaoa": {"picked": sel(res["x"]), "cost": res["best_cost"], "engine": res["engine"],
                     "layers": res["p"], "evals": res["n_evals"], "time_s": res["runtime_s"],
                     "feasible": res["best_feasible"], "is_optimal": res["is_optimal"],
                     "rank": res["rank_among_feasible"], "approx_ratio": res["approx_ratio"],
                     "random_approx_ratio": res["random_approx_ratio"],
                     "prob_optimal": res["prob_optimal"], "prob_feasible": res["prob_feasible"],
                     "random_prob_feasible": bf["n_feasible"] / 2 ** n,
                     "random_prob_optimal": 1 / 2 ** n,
                     "overlap": len(set(sel(res["x"])) & set(sel(bf["x"]))),
                     "cost_gap_pct": 100 * (res["best_cost"] - bf["cost"]) / abs(bf["cost"]) if bf["cost"] else 0.0,
                     "history": [round(h, 4) for h in res["history"]], **stat(res["x"]),
                     "top": [{"picked": sel(bits_of(int(i), n)), "prob": float(res["probs"][i]),
                              **prob.stats(bits_of(int(i), n))} for i in top]},
            "random": {"picked": sel(bits_of(rb, n)) if rb is not None else [],
                       "cost": float(d[rb]) if rb is not None else None,
                       **(stat(bits_of(rb, n)) if rb is not None else {})},
            "cloud": cloud,
        }
        out["sizes"].append(entry)
        q = entry["qaoa"]
        print(f"n={n:2d} k={k}  brute={entry['brute']['picked']}  qaoa={q['picked']}  "
              f"optimal={q['is_optimal']} rank={q['rank']}/{bf['n_feasible']}  AR={q['approx_ratio']:.3f} "
              f"(random {q['random_approx_ratio']:.3f})  P(opt)={q['prob_optimal']:.4f}  "
              f"t={q['time_s']:.1f}s [{q['engine']}]", flush=True)

    res_dir = ROOT / "results"
    res_dir.mkdir(exist_ok=True)
    (res_dir / "results.json").write_text(json.dumps(out, indent=1))
    (ROOT / "web" / "results.js").write_text("window.QSP_RESULTS = " + json.dumps(out) + ";\n")
    print("saved results/results.json and web/results.js")


if __name__ == "__main__":
    main()
