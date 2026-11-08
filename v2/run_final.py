import json, re
import rmlib
import pfm_opt2 as M
from pimlib import GEOM

NB = 15
OPT = json.load(open("results/opt3.json"))
ORGP = {"x8": "DDR4_8Gb_x8", "x16": "DDR4_8Gb_x16"}

def pim_lines(pn, U):
    T, ncol = 1 << pn, 1 << (NB - pn)
    return [(U*g + u) * T + t for g in range(ncol // U) for t in range(T) for u in range(U)]

res = {}
for org, g in GEOM.items():
    rmlib.FIELD_LSB.update(g["lsb"])
    rmlib.FIELD_CAP.update({"col": 7, "bg": g["bgbits"], "bank": 2, "row": 16})
    for pn in (7, 9):
        key = f"{org}_pn{pn}"; res[key] = {}
        print(f"\n=== {key}")
        print(f"{'layout':<14}{'NPU meas%':>10}{'NPU model':>10}{'PIM analyt%':>12}{'PIM proxy%':>11}")
        for name, e in OPT[key].items():
            order = e["order"]
            rmlib.check_bijection(order, NB)
            tag = re.sub(r"[^A-Za-z0-9_.]", "_", f"{key}_{name}")
            rn = rmlib.run_trace([rmlib.remap(i, order) for i in range(1 << NB)],
                                 tag + "_npu", org=ORGP[org])
            rp = rmlib.run_trace([rmlib.remap(i, order) for i in pim_lines(pn, g["U"])],
                                 tag + "_pim", org=ORGP[org])
            print(f"{name:<14}{rn['util']:10.2f}{e['model_npu']:10.2f}{e['pim_util']:12.2f}{rp['util']:11.2f}")
            res[key][name] = {"order": order, "npu_meas": rn["util"], "npu_model": e["model_npu"],
                              "pim_analytical": e["pim_util"], "pim_proxy": rp["util"]}
json.dump(res, open("results/final.json", "w"), indent=2)
print("\nwrote results/final.json")
