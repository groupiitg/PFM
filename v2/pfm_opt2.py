#!/usr/bin/env python3
"""PFM dual-view mapping optimizer, DDR4 version. 2 MB superpage = 15 line-index bits."""
import argparse, json, math, random
import pulp

NB = 15                                   # line-index bits (PA bits 6..20)
ORGS = {"x8":  {"col": 7, "bg": 2, "bank": 2, "row": 4},    # DDR4_8Gb_x8 (checked by probe2.py)
        "x16": {"col": 7, "bg": 1, "bank": 2, "row": 5}}    # 2 bank groups (extension)
LEVELS = ["col", "bg", "bank", "row"]
LAM, TIE = 0.25, 1e-4
FULL, HALF = 1.0, 2.0 / 3.0               # bus rate with tCCD_S vs tCCD_L (4/4 vs 4/6)

LLAMA3_8B = {"q_proj": (4096, 4096), "k_proj": (4096, 1024), "v_proj": (4096, 1024),
             "o_proj": (4096, 4096), "gate_proj": (4096, 14336),
             "up_proj": (4096, 14336), "down_proj": (14336, 4096)}

def pn_of(d_in):                           # lines per weight column, rounded up to 2^pn
    return math.ceil(math.log2(d_in * 2 / 64))

# ---------------- exact scores (also used to check the MILP) ----------------
def npu_score(m, cap):
    inv = sum(1 for i in range(NB) for j in range(i + 1, NB)
              if m[i] == "row" and m[j] == "col")
    f1 = 1.0 if m[0] == "bg" else 0.0
    return HALF + (FULL - HALF) * f1 - LAM * inv / (cap["row"] * cap["col"])

def pim_score(m, cap, pn):
    ub = cap["bg"] + cap["bank"]
    g2 = 1.0 if m[pn] == "bg" else 0.0
    rowbad = sum(1 for j in range(pn + ub) if m[j] == "row")
    return HALF + (FULL - HALF) * g2 - LAM * rowbad / cap["row"]

def objective(m, cap, pn, alpha):
    return (npu_score(m, cap) + alpha * pim_score(m, cap, pn)
            + TIE * sum(j for j, l in m.items() if l == "row"))

# ------------------------------- MILP (PuLP) --------------------------------
def solve(cap, pn, alpha):
    ub = cap["bg"] + cap["bank"]
    P = pulp.LpProblem("pfm", pulp.LpMaximize)
    x = {(l, j): pulp.LpVariable(f"x_{l}_{j}", cat="Binary") for l in LEVELS for j in range(NB)}
    for j in range(NB):                                        # structural
        P += pulp.lpSum(x[l, j] for l in LEVELS) == 1
    for l in LEVELS:                                           # capacity
        P += pulp.lpSum(x[l, j] for j in range(NB)) == cap[l]
    z = {(i, j): pulp.LpVariable(f"z_{i}_{j}", 0, 1) for i in range(NB) for j in range(i + 1, NB)}
    for (i, j), v in z.items():                                # z_ij >= row_i AND col_j
        P += v >= x["row", i] + x["col", j] - 1
    sN = HALF + (FULL - HALF) * x["bg", 0] - LAM * pulp.lpSum(z.values()) / (cap["row"] * cap["col"])
    sP = (HALF + (FULL - HALF) * x["bg", pn]
          - LAM * pulp.lpSum(x["row", j] for j in range(pn + ub)) / cap["row"])
    tie = TIE * pulp.lpSum(j * x["row", j] for j in range(NB))
    P += sN + alpha * sP + tie
    P.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[P.status] == "Optimal", pulp.LpStatus[P.status]
    m = {j: next(l for l in LEVELS if pulp.value(x[l, j]) > 0.5) for j in range(NB)}
    return m, pulp.value(P.objective)

# ------------------------- layouts and formatting ----------------------------
def layout(order, cap):
    m, j = {}, 0
    for l, n in order:
        for _ in range(n):
            m[j] = l; j += 1
    assert j == NB and all(sum(v == l for v in m.values()) == cap[l] for l in LEVELS)
    return m

def fixed_layouts(cap):
    return {"NPU-friendly": layout([("bg", cap["bg"]), ("col", cap["col"]),
                                    ("bank", cap["bank"]), ("row", cap["row"])], cap),
            "PIM-friendly": layout([("col", cap["col"]), ("bg", cap["bg"]),
                                    ("bank", cap["bank"]), ("row", cap["row"])], cap)}

def to_order(m):                           # run-length encode, LSB -> MSB
    out = []
    for j in range(NB):
        if out and out[-1][0] == m[j]: out[-1][1] += 1
        else: out.append([m[j], 1])
    return out

def fmt(m):
    return " ".join(f"{l}{n}" for l, n in to_order(m))

def random_best(cap, pn, alpha, trials, seed=0):
    bits = [l for l in LEVELS for _ in range(cap[l])]
    rng, best = random.Random(seed), -1e9
    for _ in range(trials):
        rng.shuffle(bits)
        best = max(best, objective(dict(enumerate(bits)), cap, pn, alpha))
    return best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="x8", choices=list(ORGS))
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--trials", type=int, default=100000)
    a = ap.parse_args()
    cap = ORGS[a.org]

    groups = {}
    for name, (d_in, _) in LLAMA3_8B.items():
        groups.setdefault(pn_of(d_in), []).append(name)

    out = {}
    for pn, names in sorted(groups.items()):
        print(f"\n=== org={a.org}  pn={pn} (column = 2^{pn} lines)  tensors: {', '.join(names)}")
        m, obj = solve(cap, pn, a.alpha)
        assert abs(obj - objective(m, cap, pn, a.alpha)) < 1e-6, "MILP/exact mismatch"
        rb = random_best(cap, pn, a.alpha, a.trials)
        assert obj >= rb - 1e-9, "random search beat the solver"
        print(f"alpha={a.alpha}: solver {obj:.4f} | best of {a.trials} random maps {rb:.4f}")

        cands = dict(fixed_layouts(cap)); cands["PFM-joint"] = m
        print(f"{'layout':<14}{'model NPU':>10}{'model PIM':>10}   order LSB->MSB")
        for n, mm in cands.items():
            print(f"{n:<14}{npu_score(mm, cap):10.3f}{pim_score(mm, cap, pn):10.3f}   {fmt(mm)}")

        print("\n alpha sweep:")
        print(f" {'alpha':>6}{'model NPU':>10}{'model PIM':>10}   order LSB->MSB")
        prev = None
        for al in [0, 0.25, 0.5, 1, 2, 5]:
            ma, _ = solve(cap, pn, al)
            sc = (round(npu_score(ma, cap), 6), round(pim_score(ma, cap, pn), 6))
            print(f" {al:6.2f}{sc[0]:10.3f}{sc[1]:10.3f}   {fmt(ma)}"
                  f"{'   <- scores changed' if prev and sc != prev else ''}")
            prev = sc
        out[str(pn)] = {"tensors": names, "alpha": a.alpha, "order": to_order(m)}

    with open(f"results/mappings_{a.org}.json", "w") as f:
        json.dump({"org": a.org, "cap": cap, "nbits": NB, "mappings": out}, f, indent=2)
    print(f"\nwrote results/mappings_{a.org}.json")

if __name__ == "__main__":
    main()
