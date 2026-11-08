import json, csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = json.load(open("results/final.json"))
SHAPES = {"q_proj":(4096,4096,16),"k_proj":(4096,1024,4),"v_proj":(4096,1024,4),
          "o_proj":(4096,4096,16),"gate_proj":(4096,14336,56),"up_proj":(4096,14336,56),
          "down_proj":(14336,4096,56)}
LAY = ["NPU-friendly","PIM-friendly","PFM a=0.25","PFM a=2.0"]
pn_of = lambda t: 9 if t == "down_proj" else 7

rows = []
for org in ("x8","x16"):
    for lay in LAY:
        wn = wp = tot = 0
        for t,(di,do,pg) in SHAPES.items():
            e = R[f"{org}_pn{pn_of(t)}"][lay]
            rows.append([org,t,f"{di}x{do}",pg,lay,round(e["npu_meas"],2),round(e["pim_analytical"],2)])
            wn += pg*e["npu_meas"]; wp += pg*e["pim_analytical"]; tot += pg
        rows.append([org,"ALL(page-weighted)","",tot,lay,round(wn/tot,2),round(wp/tot,2)])
with open("results/shape_table.csv","w",newline="") as f:
    w = csv.writer(f); w.writerow(["org","tensor","shape","pages","layout","npu_util_meas","pim_util_analytical"]); w.writerows(rows)

print(f"{'org':<5}{'layout':<14}{'NPU% (weighted)':>16}{'PIM% analytical (weighted)':>28}")
for r in rows:
    if r[1].startswith("ALL"): print(f"{r[0]:<5}{r[4]:<14}{r[5]:>16}{r[6]:>28}")

# Fig 1: grouped bars, x8, pn=7 and pn=9
fig, ax = plt.subplots(1, 2, figsize=(11,4), sharey=True)
for a, pn in zip(ax, (7,9)):
    d = R[f"x8_pn{pn}"]; x = range(len(LAY)); w = 0.38
    a.bar([i-w/2 for i in x], [d[l]["npu_meas"] for l in LAY], w, label="NPU seq. read (Ramulator)")
    a.bar([i+w/2 for i in x], [d[l]["pim_analytical"] for l in LAY], w, label="PIM all-bank (analytical)")
    a.set_xticks(list(x)); a.set_xticklabels(["NPU-\nfriendly","PIM-\nfriendly","PFM\nα=0.25","PFM\nα=2"])
    a.set_title(f"DDR4 x8, pn={pn} ({'6 tensors' if pn==7 else 'down_proj'})"); a.set_ylim(0,105)
ax[0].set_ylabel("Bandwidth utilization (%)"); ax[0].legend(loc="upper center", fontsize=8)
plt.tight_layout(); plt.savefig("results/fig_util_x8.png", dpi=200)

# Fig 2: NPU vs PIM trade-off, all layouts
fig, ax = plt.subplots(figsize=(6,5))
for key, mk in (("x8_pn7","o"),("x8_pn9","s"),("x16_pn7","^"),("x16_pn9","v")):
    for l,e in R[key].items():
        ax.scatter(e["npu_meas"], e["pim_analytical"], marker=mk, c="C1" if l.startswith("PFM") else "C0")
ax.set_xlabel("NPU utilization, measured (%)"); ax.set_ylabel("PIM utilization, analytical (%)")
ax.set_title("Trade-off (orange = PFM, blue = fixed layouts)\no/s = x8 pn7/pn9, ^/v = x16 pn7/pn9", fontsize=9)
ax.grid(alpha=.3); plt.tight_layout(); plt.savefig("results/fig_tradeoff.png", dpi=200)
print("\nwrote results/shape_table.csv, fig_util_x8.png, fig_tradeoff.png")
