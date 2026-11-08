import json
import rmlib
rmlib.FIELD_LSB.update({"col": 6, "bg": 13, "bank": 14, "row": 16})
rmlib.FIELD_CAP.update({"col": 7, "bg": 1, "bank": 2, "row": 16})
from rmlib import remap, check_bijection, run_trace
import pfm_opt2 as M

NB, cap = 15, M.ORGS["x16"]
fx = M.fixed_layouts(cap)

def pim_lines(pn, U=8):
    T, ncol = 1 << pn, 1 << (NB - pn)
    return [(U*g + u) * T + t for g in range(ncol // U) for t in range(T) for u in range(U)]

out = {}
for pn in (7, 9):
    L = {"NPU-friendly": M.to_order(fx["NPU-friendly"]),
         "PIM-friendly": M.to_order(fx["PIM-friendly"])}
    for al in (0.25, 2.0):
        m, _ = M.solve(cap, pn, al)
        L[f"PFM a={al}"] = M.to_order(m)
    print(f"\n=== x16 pn={pn}")
    print(f"{'layout':<14}{'NPU util%':>10}{'PIM util%':>10}{'PIM rowconf':>12}   order")
    out[pn] = {}
    for name, order in L.items():
        check_bijection(order, NB)
        rn = run_trace([remap(i, order) for i in range(1 << NB)], f"x16_pn{pn}_{name}_npu".replace(" ", "_"), org="DDR4_8Gb_x16")
        rp = run_trace([remap(i, order) for i in pim_lines(pn)], f"x16_pn{pn}_{name}_pim".replace(" ", "_"), org="DDR4_8Gb_x16")
        print(f"{name:<14}{rn['util']:10.2f}{rp['util']:10.2f}{rp['row_conflicts']:12d}   "
              + " ".join(f"{l}{n}" for l, n in order))
        out[pn][name] = {"order": order, "npu": rn, "pim": rp}
json.dump(out, open("results/measured_x16.json", "w"), indent=2)
print("\nwrote results/measured_x16.json")
