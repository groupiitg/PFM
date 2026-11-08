import json
from rmlib import remap, check_bijection, run_trace
import pfm_opt2 as M

NB = 15
cfg = json.load(open("results/mappings_x8.json"))
cap = cfg["cap"]

FIXED = {"NPU-friendly": [("bg",2),("col",7),("bank",2),("row",4)],
         "PIM-friendly": [("col",7),("bg",2),("bank",2),("row",4)]}

def to_map(order):
    m, j = {}, 0
    for l, n in order:
        for _ in range(n):
            m[j] = l; j += 1
    return m

def npu_lines():
    return list(range(1 << NB))

def pim_lines(pn):
    T, ncol = 1 << pn, 1 << (NB - pn)
    return [(16*g + u) * T + t
            for g in range(ncol // 16) for t in range(T) for u in range(16)]

results = {}
for pn_s, entry in cfg["mappings"].items():
    pn = int(pn_s)
    layouts = dict(FIXED); layouts["PFM-joint"] = entry["order"]
    print(f"\n=== pn={pn}  tensors: {', '.join(entry['tensors'])}")
    print(f"{'layout':<14}{'NPU util%':>10}{'NPU model':>10}{'PIM util%':>10}{'PIM model':>10}"
          f"{'PIM rowconf':>12}")
    results[pn_s] = {}
    for name, order in layouts.items():
        check_bijection(order, NB)
        m = to_map(order)
        tag = f"x8_pn{pn}_{name.replace('-', '_')}"
        rn = run_trace([remap(l, order) for l in npu_lines()], tag + "_npu")
        rp = run_trace([remap(l, order) for l in pim_lines(pn)], tag + "_pim")
        print(f"{name:<14}{rn['util']:10.2f}{M.npu_score(m, cap):10.3f}"
              f"{rp['util']:10.2f}{M.pim_score(m, cap, pn):10.3f}{rp['row_conflicts']:12d}")
        results[pn_s][name] = {"order": order, "npu": rn, "pim": rp,
                               "model_npu": M.npu_score(m, cap),
                               "model_pim": M.pim_score(m, cap, pn)}

json.dump(results, open("results/measured_x8.json", "w"), indent=2)
print("\nwrote results/measured_x8.json")
