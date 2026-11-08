"""Run the layouts through the PFMRegionMapper added to Ramulator 2.0.

Part 1 (equivalence): for every layout in results/final.json, replay the plain
(un-remapped) NPU and PIM traces with PFMRegionMapper doing the mapping inside
Ramulator, and the pre-remapped traces with the stock RoBaRaCoCh mapper. Both
must give identical cycle counts.

Part 2 (several regions in one run): one NPU stream over a block of 2 MB
regions that hold different LLaMA3-8B matrices, each region with its own layout.

Output: results/region_mapper.json
"""
import json
import os
import re

import rmlib
import ramulator
from pimlib import GEOM

NB = 15
ORGP = {"x8": "DDR4_8Gb_x8", "x16": "DDR4_8Gb_x16"}
FINAL = json.load(open("results/final.json"))
os.makedirs("traces", exist_ok=True)


def layout_str(order):
    return " ".join(f"{l}{n}" for l, n in order)


def run(addrs, tag, org, mapper):
    assert len(set(addrs)) == len(addrs), "duplicate addresses in trace"
    path = f"traces/{tag}.trace"
    with open(path, "w") as f:
        for a in addrs:
            f.write(f"0 {a}\n")
    n = len(addrs)
    fe = ramulator.frontend.SimpleO3(
        clock_ratio=8, traces=[path], num_expected_insts=n,
        translation=ramulator.translation.NoTranslation(max_addr=2147483648))
    dram = ramulator.dram.DDR4(org_preset=org, timing_preset="DDR4_2400R", rank=1, verbose=False)
    ctrl = ramulator.controller.GenericDDR(
        dram=dram, scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(),
        row_policy=ramulator.row_policy.Open(), addr_mapper=mapper)
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=3, controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
    sim = ramulator.Simulation(fe, mem)
    sim.run()
    c = sim.stats["memory_system"]["controller"]
    # the trace frontend may wrap around and issue a few extra reads at the very end
    served = c["num_read_reqs_served"]
    assert n <= served <= n + 16 and c["num_read_reqs_forwarded"] == 0, \
        f"{served}/{n} reads reached DRAM"
    os.remove(path)
    return {"cycles": c["cycles"], "served": served, "util": 100.0 * n * 4 / c["cycles"],
            "row_hits": c["row_hits"], "row_conflicts": c["row_conflicts"],
            "mapper": c.get("addr_mapper", {})}


def pim_lines(pn, U):
    T, ncol = 1 << pn, 1 << (NB - pn)
    return [(U * g + u) * T + t for g in range(ncol // U) for t in range(T) for u in range(U)]


out = {"equivalence": {}, "multi_region": {}}

# ---------------- Part 1: equivalence with the pre-remap ----------------
print("Part 1: PFMRegionMapper inside Ramulator vs Python pre-remap + RoBaRaCoCh")
print(f"{'case':<9}{'layout':<14}{'NPU mapper%':>12}{'NPU remap%':>11}{'saved%':>8}"
      f"{'proxy mapper%':>14}{'proxy remap%':>13}  same cycles")
all_same = True
for org, g in GEOM.items():
    rmlib.FIELD_LSB.update(g["lsb"])
    rmlib.FIELD_CAP.update({"col": 7, "bg": g["bgbits"], "bank": 2, "row": 16})
    for pn in (7, 9):
        key = f"{org}_pn{pn}"
        out["equivalence"][key] = {}
        for name, e in FINAL[key].items():
            order = e["order"]
            tag = re.sub(r"[^A-Za-z0-9_.]", "_", f"rm_{key}_{name}")
            pfm = ramulator.addr_mapper.PFMRegionMapper(layouts=[layout_str(order)], region_layout=[0])
            res = {}
            for kind, lines in (("npu", list(range(1 << NB))), ("pim", pim_lines(pn, g["U"]))):
                a = run([l * 64 for l in lines], f"{tag}_{kind}_m", ORGP[org], pfm)
                b = run([rmlib.remap(l, order) for l in lines], f"{tag}_{kind}_r", ORGP[org],
                        ramulator.addr_mapper.RoBaRaCoCh())
                res[kind] = {"mapper": a, "remap": b, "same": a["cycles"] == b["cycles"]}
            same = res["npu"]["same"] and res["pim"]["same"]
            all_same &= same
            print(f"{key:<9}{name:<14}{res['npu']['mapper']['util']:12.2f}{res['npu']['remap']['util']:11.2f}"
                  f"{e['npu_meas']:8.2f}{res['pim']['mapper']['util']:14.2f}{res['pim']['remap']['util']:13.2f}"
                  f"  {'yes' if same else 'NO'}")
            out["equivalence"][key][name] = {
                "layout": layout_str(order),
                "npu_util_mapper": res["npu"]["mapper"]["util"], "npu_util_remap": res["npu"]["remap"]["util"],
                "npu_util_saved": e["npu_meas"], "npu_cycles": res["npu"]["mapper"]["cycles"],
                "pim_proxy_util_mapper": res["pim"]["mapper"]["util"],
                "pim_proxy_util_remap": res["pim"]["remap"]["util"],
                "identical": same}
print("all identical:", all_same)
out["all_identical"] = all_same

# ---------------- Part 2: several regions with different layouts in one run ----------------
# A block of 8 regions (16 MB) laid out like a slice of one decoder layer:
# 6 regions of d_in=4096 matrices and 2 regions of down_proj (d_in=14336).
print("\nPart 2: one NPU stream over 8 regions, each region with its own layout (DDR4 x8)")
rmlib.FIELD_LSB.update(GEOM["x8"]["lsb"])
rmlib.FIELD_CAP.update({"col": 7, "bg": 2, "bank": 2, "row": 16})
regions = ["pn7"] * 6 + ["pn9"] * 2
lines = list(range(len(regions) << NB))
plans = {
    "NPU-friendly everywhere": {"pn7": "NPU-friendly", "pn9": "NPU-friendly"},
    "PIM-friendly everywhere": {"pn7": "PIM-friendly", "pn9": "PIM-friendly"},
    "PFM a=0.25 per region": {"pn7": "PFM a=0.25", "pn9": "PFM a=0.25"},
    "PFM a=2.0 per region": {"pn7": "PFM a=2.0", "pn9": "PFM a=2.0"},
}
print(f"{'plan':<26}{'NPU util%':>10}{'regions mapped':>16}   layouts")
for plan, pick in plans.items():
    lay = {cls: layout_str(FINAL[f"x8_{cls}"][pick[cls]]["order"]) for cls in ("pn7", "pn9")}
    table = sorted(set(lay.values()))
    ids = [table.index(lay[cls]) for cls in regions]
    pfm = ramulator.addr_mapper.PFMRegionMapper(layouts=table, region_layout=ids)
    r = run([l * 64 for l in lines], re.sub(r"[^A-Za-z0-9]", "_", "multi_" + plan), "DDR4_8Gb_x8", pfm)
    mapped = r["mapper"].get("region_mapped", "?")
    print(f"{plan:<26}{r['util']:10.2f}{str(mapped):>16}   {len(table)} layout(s), region_layout={ids}")
    out["multi_region"][plan] = {"util": r["util"], "cycles": r["cycles"], "layouts": table,
                                 "region_layout": ids, "region_mapped": mapped,
                                 "reads": len(lines)}

json.dump(out, open("results/region_mapper.json", "w"), indent=2)
print("\nwrote results/region_mapper.json")
