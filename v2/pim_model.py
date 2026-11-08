import json
import rmlib
from collections import defaultdict
import pfm_opt2 as M

CCDL, RCD, RP = 6, 16, 16
NB = 15
GEOM = {  # field LSBs, bank-group bits, units, caps
  "x8":  dict(lsb={"col":6,"bg":13,"bank":15,"row":17}, bgbits=2, U=16, cap=M.ORGS["x8"]),
  "x16": dict(lsb={"col":6,"bg":13,"bank":14,"row":16}, bgbits=1, U=8,  cap=M.ORGS["x16"]),
}

def pim_util(order, pn, g):
    rmlib.FIELD_LSB.update(g["lsb"])
    rmlib.FIELD_CAP.update({"col":7,"bg":g["bgbits"],"bank":2,"row":16})
    U, T, ncol = g["U"], 1 << pn, 1 << (NB - pn)
    L = g["lsb"]; bgm = (1 << g["bgbits"]) - 1
    openrow, total, steps = {}, 0, 0
    for grp in range(ncol // U):
        for t in range(T):
            cost = defaultdict(int)
            for u in range(U):
                ha = rmlib.remap((U*grp + u) * T + t, order)
                bank = ((ha >> L["bg"]) & bgm, (ha >> L["bank"]) & 3)
                row = ha >> L["row"]
                c = CCDL
                if bank not in openrow: c += RCD
                elif openrow[bank] != row: c += RP + RCD
                openrow[bank] = row
                cost[bank] += c
            total += max(cost.values()); steps += 1
    return 100.0 * steps * CCDL / total

PAGES = {"q_proj":16,"k_proj":4,"v_proj":4,"o_proj":16,"gate_proj":56,"up_proj":56,"down_proj":56}
out = {}
for org, g in GEOM.items():
    fx = M.fixed_layouts(g["cap"])
    for pn in (7, 9):
        L = {"NPU-friendly": M.to_order(fx["NPU-friendly"]), "PIM-friendly": M.to_order(fx["PIM-friendly"])}
        if org == "x8":
            L["PFM-joint"] = json.load(open("results/mappings_x8.json"))["mappings"][str(pn)]["order"]
        else:
            for al in (0.25, 2.0): L[f"PFM a={al}"] = M.to_order(M.solve(g["cap"], pn, al)[0])
        print(f"\n=== {org} pn={pn}  (analytical all-bank PIM util %)")
        out[f"{org}_pn{pn}"] = {}
        for n, o in L.items():
            u = pim_util(o, pn, g); out[f"{org}_pn{pn}"][n] = u
            print(f"{n:<14}{u:8.2f}")
json.dump({"pim_analytical": out, "pages": PAGES}, open("results/pim_analytical.json", "w"), indent=2)
