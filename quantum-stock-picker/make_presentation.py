"""Build docs/Quantum_Stock_Picker_Presentation.pdf (16:9 slides) from results + figures."""
import json
from pathlib import Path
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor
from reportlab.lib.utils import ImageReader, simpleSplit

ROOT = Path(__file__).parent
FIG = ROOT / "results" / "figures"
R = json.loads((ROOT / "results" / "results.json").read_text())
S = {e["n"]: e for e in R["sizes"]}
W, H = 960, 540
INK, MUT, QC, CC, BG, LINE = (HexColor(c) for c in ("#15222c", "#5b6b77", "#e0870b", "#0f7c8a", "#eef2f4", "#d3dce2"))
out = ROOT / "docs" / "Quantum_Stock_Picker_Presentation.pdf"
c = canvas.Canvas(str(out), pagesize=(W, H))
c.setTitle("Quantum Stock Picker"); c.setAuthor("Quantum Stock Picker project")
page = [0]


def slide(title, sub=None):
    if page[0]:
        c.showPage()
    page[0] += 1
    c.setFillColor(BG); c.rect(0, 0, W, H, stroke=0, fill=1)
    c.setFillColor(QC); c.rect(0, H - 8, W, 8, stroke=0, fill=1)
    c.setFillColor(INK); c.setFont("Times-Bold", 32); c.drawString(48, H - 70, title)
    if sub:
        c.setFillColor(MUT); c.setFont("Helvetica", 15); c.drawString(48, H - 94, sub)
    c.setFillColor(MUT); c.setFont("Helvetica", 10); c.drawRightString(W - 40, 22, f"Quantum Stock Picker  |  {page[0]}")


def text(x, y, s, size=16, width=400, color=INK, font="Helvetica", lead=1.35):
    c.setFillColor(color); c.setFont(font, size)
    for line in simpleSplit(s, font, size, width):
        c.drawString(x, y, line); y -= size * lead
    return y


def bullets(x, y, items, size=17, width=420, gap=10):
    for it in items:
        c.setFillColor(QC); c.circle(x + 4, y + size * .32, 3.2, stroke=0, fill=1)
        y = text(x + 20, y, it, size, width - 20) - gap
    return y


def img(path, x, y, w, h=None):
    ir = ImageReader(str(path)); iw, ih = ir.getSize()
    h = h or w * ih / iw
    if h > 400 and not False:
        pass
    c.drawImage(ir, x, y, w, h, mask="auto"); return h


def fit(path, x, y, maxw, maxh):
    iw, ih = ImageReader(str(path)).getSize()
    s = min(maxw / iw, maxh / ih)
    c.drawImage(ImageReader(str(path)), x, y, iw * s, ih * s, mask="auto")


# 1 title
slide("Quantum Stock Picker")
text(48, H - 150, "Choosing a few stocks with high profit and low risk using QAOA in Qiskit", 24, 760, MUT)
text(48, 150, "Encode  >  Solve with QAOA  >  Check against brute force  >  Scale 8 to 20 stocks  >  Visualise", 17, 860, INK)
text(48, 100, "Synthetic data, classical simulation of the quantum circuits. Not investment advice.", 13, 860, MUT)

# 2 problem
slide("The problem", "Many stocks, pick a few")
bullets(48, H - 140, [
    "We want a small basket of k stocks with high expected profit and low risk.",
    "Risk depends on how stocks move together, so the best basket is not simply the k highest-return names.",
    "Each stock is in or out, so n stocks give 2^n possible baskets: 1,048,576 at n = 20, and a doubling for every extra stock.",
    "Idea: let a quantum circuit favour good baskets, then measure it.",
], 18, 520)
c.setFillColor(HexColor("#ffffff")); c.setStrokeColor(LINE); c.roundRect(640, 150, 270, 230, 10, stroke=1, fill=1)
text(660, 350, "Number of baskets", 14, 230, MUT)
for i, (n, v) in enumerate([(8, "256"), (12, "4,096"), (16, "65,536"), (20, "1,048,576")]):
    text(660, 318 - i * 38, f"{n} stocks", 16, 100, INK); c.setFont("Helvetica-Bold", 16); c.drawRightString(895, 318 - i * 38, v)

# 3 encoding
slide("Setting up the problem", "One qubit per stock: |1> = picked, |0> = not picked")
text(48, H - 140, "Cost of a basket x (lower is better)", 17, 500, MUT)
c.setFillColor(INK); c.setFont("Courier-Bold", 19)
c.drawString(48, H - 175, "C(x) = q * x'Sx  -  m'x  +  L * (sum(x) - k)^2")
bullets(48, H - 215, [
    "x'Sx: risk (covariance S of returns), weighted by risk aversion q.",
    "m'x: expected profit of the picked stocks.",
    "L (sum(x) - k)^2: penalty unless exactly k stocks are picked.",
    "Substitute x = (1 - Z)/2 to get an Ising Hamiltonian: single-qubit Z terms (profit and penalty) plus ZZ couplings (how stocks interact).",
    "The lowest-energy bitstring is the best basket.",
], 16, 860, 6)

# 4 circuit
slide("Solving it with QAOA", "Alternate a cost layer and a mixer layer, p times")
if (FIG / "circuit_n4.png").exists():
    fit(FIG / "circuit_n4.png", 40, 235, 880, 200)
text(48, 220, "Example: 3 stocks, p = 1.  H gates make an equal mix of all baskets; RZ/RZZ gates (angle gamma) add the cost; RX gates (angle beta) mix.", 14, 860, MUT)
bullets(48, 160, [
    "A classical optimiser (COBYLA) tunes gamma and beta to lower the average cost; our runs use p = 3.",
    "Measure 2048 times; keep the cheapest valid basket.",
], 15, 860, 4)

# 5 results n=8
e = S[8]; q = e["qaoa"]
slide("Result: 8 stocks", "Picking 3 of 8 (56 valid baskets)")
fit(FIG / "stocks_n8.png", 40, 70, 560, 380)
bullets(630, H - 150, [
    f"QAOA picked {', '.join(q['picked'])}.",
    f"Exhaustive search: {', '.join(e['brute']['picked'])}." + ("  Same answer." if q["is_optimal"] else ""),
    f"Profit {100*q['profit']:.1f}%, risk {100*q['risk']:.1f}%.",
    f"A single shot hits the best basket {100*q['prob_optimal']:.1f}% of the time (random: {100*q['random_prob_optimal']:.2f}%).",
], 15, 300, 8)

# 6 verification
slide("Checking the answer", "Slow way = score every one of the 2^n baskets")
hdr = ["Stocks", "Baskets", "Found best?", "Rank", "Appr. ratio", "Random", "P(best)/shot"]
x0, y0, cw = 48, H - 150, [90, 130, 130, 110, 120, 110, 150]
c.setFont("Helvetica-Bold", 14); c.setFillColor(MUT)
xx = x0
for h_, w_ in zip(hdr, cw):
    c.drawString(xx, y0, h_); xx += w_
for i, n in enumerate(sorted(S)):
    e = S[n]; q = e["qaoa"]; yy = y0 - 40 - i * 40
    vals = [str(n), f"{e['n_states']:,}", "yes" if q["is_optimal"] else "no", f"{q['rank']} of {e['n_feasible']:,}",
            f"{q['approx_ratio']:.3f}", f"{q['random_approx_ratio']:.3f}", f"{100*q['prob_optimal']:.3f}%"]
    c.setFillColor(INK if q["is_optimal"] else QC); c.setFont("Helvetica", 15)
    xx = x0
    for v, w_ in zip(vals, cw):
        c.drawString(xx, yy, v); xx += w_
    c.setStrokeColor(LINE); c.line(x0, yy - 12, W - 48, yy - 12)
text(48, 150, "Approximation ratio = (worst cost - QAOA average cost) / (worst - best). 1 is perfect; 'Random' is a uniformly random guess.", 13, 860, MUT)
text(48, 110, "Rank = position of the QAOA pick among all valid baskets (1 = best possible).", 13, 860, MUT)

# 7 scaling
slide("Making it bigger: 12, 16 and 20 stocks")
fit(FIG / "scaling.png", 40, 175, 880, 270)
s16, s20 = S.get(16), S.get(20)
msg = "QAOA's average cost stays well above random guessing, but the chance that one shot is exactly the best basket falls quickly as the search space doubles per stock."
text(48, 150, msg, 15, 860, INK)
if s20:
    text(48, 100, f"At 20 stocks the QAOA pick is rank {s20['qaoa']['rank']} of {s20['n_feasible']:,} valid baskets, within {abs(s20['qaoa']['cost_gap_pct']):.1f}% of the best cost.", 15, 860, MUT)

# 8 chart of risk vs profit
big = S[max(S)]
slide(f"Risk against profit: {big['n']} stocks", "Orange = picked by our quantum method, teal ring = best possible")
fit(FIG / f"stocks_n{big['n']}.png", 30, 60, 450, 380)
fit(FIG / f"portfolios_n{big['n']}.png", 490, 60, 440, 380)

# 9 takeaways
slide("What we learned")
bullets(48, H - 120, [
    "The formulation works: a QUBO / Ising cost with a cardinality penalty puts the best basket at the lowest energy.",
    "At 8 and 12 qubits QAOA found the exact best portfolio.",
    "As n grows, the chance of hitting the optimum in one shot drops, yet the QAOA picks stay close to optimal in cost and risk/profit.",
    "Caveats: noise-free simulation, synthetic data, fixed shallow depth (p = 3), plain X mixer. Real hardware and bigger n need better mixers (e.g. constraint-preserving) and warm starts.",
    "Next steps: real market data, constraint-preserving mixers, deeper p, noisy-device simulation.",
], 17, 860, 12)

# 10 run it
slide("Run it yourself", "github-ready project")
for i, line in enumerate(["pip install -r requirements.txt", "python run_experiments.py", "python make_figures.py", "python make_presentation.py", "open web/index.html"]):
    c.setFillColor(INK); c.setFont("Courier-Bold", 20); c.drawString(60, H - 160 - i * 44, "$ " + line)
text(60, 170, "Folders: src/qsp (code), tests, results, web (HTML/CSS frontend), docs.", 15, 800, MUT)
c.save(); print("saved", out)
