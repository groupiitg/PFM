import json, random, pulp
import pfm_opt2 as M
from pimlib import pim_util, GEOM
NB, LEVELS = M.NB, M.LEVELS

def win(cap, pn):
    ub = cap["bg"] + cap["bank"]; return ub, list(range(pn, pn + ub))

def pim_score(m, cap, pn):
    ub, w = win(cap, pn)
    k = sum(m[j] in ("bg", "bank") for j in w); r = sum(m[j] == "row" for j in w)
    return 2.0 ** -(ub - k) * 0.5 ** r

def objective(m, cap, pn, a):
    return (M.npu_score(m, cap) + a * pim_score(m, cap, pn)
            + M.TIE * sum(j for j, l in m.items() if l == "row"))

def solve(cap, pn, a):
    ub, w = win(cap, pn)
    P = pulp.LpProblem("pfm3", pulp.LpMaximize)
    x = {(l, j): pulp.LpVariable(f"x_{l}_{j}", cat="Binary") for l in LEVELS for j in range(NB)}
    for j in range(NB): P += pulp.lpSum(x[l, j] for l in LEVELS) == 1
    for l in LEVELS: P += pulp.lpSum(x[l, j] for j in range(NB)) == cap[l]
    z = {(i, j): pulp.LpVariable(f"z_{i}_{j}", 0, 1) for i in range(NB) for j in range(i + 1, NB)}
    for (i, j), v in z.items(): P += v >= x["row", i] + x["col", j] - 1
    pairs = [(k, r) for k in range(ub + 1) for r in range(ub + 1 - k)]
    y = {p: pulp.LpVariable(f"y_{p[0]}_{p[1]}", cat="Binary") for p in pairs}
    P += pulp.lpSum(y.values()) == 1
    P += pulp.lpSum(k * y[k, r] for k, r in pairs) == pulp.lpSum(x[l, j] for l in ("bg", "bank") for j in w)
    P += pulp.lpSum(r * y[k, r] for k, r in pairs) == pulp.lpSum(x["row", j] for j in w)
    sN = M.HALF + (M.FULL - M.HALF) * x["bg", 0] - M.LAM * pulp.lpSum(z.values()) / (cap["row"] * cap["col"])
    sP = pulp.lpSum(2.0 ** -(ub - k) * 0.5 ** r * y[k, r] for k, r in pairs)
    P += sN + a * sP + M.TIE * pulp.lpSum(j * x["row", j] for j in range(NB))
    P.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[P.status] == "Optimal"
    return {j: next(l for l in LEVELS if pulp.value(x[l, j]) > .5) for j in range(NB)}, pulp.value(P.objective)

out = {}
for org, g in GEOM.items():
    cap = g["cap"]; fx = M.fixed_layouts(cap)
    for pn in (7, 9):
        print(f"\n=== {org} pn={pn}")
        print(f"{'layout':<16}{'mNPU':>6}{'mPIM':>7}{'PIM util% (analytical)':>24}   order")
        L = dict(fx)
        for a in (0.25, 1.0, 2.0):
            m, obj = solve(cap, pn, a)
            assert abs(obj - objective(m, cap, pn, a)) < 1e-6, "MILP/exact mismatch"
            rb = max(objective(dict(enumerate(random.Random(s).sample(
                 [l for l in LEVELS for _ in range(cap[l])], NB))), cap, pn, a) for s in range(20000))
            assert obj >= rb - 1e-9, "random beat solver"
            L[f"PFM a={a}"] = m
        out[f"{org}_pn{pn}"] = {}
        for n, m in L.items():
            order = M.to_order(m); u = pim_util(order, pn, g)
            print(f"{n:<16}{M.npu_score(m,cap):6.2f}{pim_score(m,cap,pn):7.3f}{u:24.2f}   {M.fmt(m)}")
            out[f"{org}_pn{pn}"][n] = {"order": order, "pim_util": u,
                                       "model_npu": M.npu_score(m, cap)}
json.dump(out, open("results/opt3.json", "w"), indent=2)
