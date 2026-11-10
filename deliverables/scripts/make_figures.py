#!/usr/bin/env python3
"""Make all figures (PDF + PNG) and LaTeX tables from the result files in v2/results."""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RES = os.path.join(ROOT, "..", "v2", "results")
FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(ROOT, "tables")
os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "axes.axisbelow": True, "grid.alpha": 0.3, "grid.linewidth": 0.5,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "pdf.fonttype": 42,
})

LAY = ["NPU-friendly", "PIM-friendly", "PFM a=0.25", "PFM a=2.0"]
LABEL = {"NPU-friendly": "NPU-friendly", "PIM-friendly": "PIM-friendly",
         "PFM a=0.25": r"PFM $\alpha$=0.25", "PFM a=2.0": r"PFM $\alpha$=2",
         "PFM a=1.0": r"PFM $\alpha$=1"}
COLOR = {"NPU-friendly": "#3b6fb6", "PIM-friendly": "#e08a2c",
         "PFM a=0.25": "#3a9a5b", "PFM a=2.0": "#8e5bb5", "PFM a=1.0": "#b0457a"}
CASES = [("x8_pn7", "x8, $d_{in}$=4096"), ("x8_pn9", "x8, $d_{in}$=14336"),
         ("x16_pn7", "x16, $d_{in}$=4096"), ("x16_pn9", "x16, $d_{in}$=14336")]
COL_W, FULL_W = 3.5, 7.16

R = json.load(open(os.path.join(RES, "final.json")))
SHAPE_ROWS = list(csv.DictReader(open(os.path.join(RES, "shape_table.csv"))))
MEAS_X8 = json.load(open(os.path.join(RES, "measured_x8.json")))


def save(fig, name):
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=300)
    plt.close(fig)


def grouped(ax, layouts, metric, ylabel, title):
    n = len(layouts)
    w = 0.8 / n
    for i, lay in enumerate(layouts):
        xs = [c + (i - (n - 1) / 2) * w for c in range(len(CASES))]
        vals = [R[k][lay][metric] for k, _ in CASES]
        bars = ax.bar(xs, vals, w, color=COLOR[lay], label=LABEL[lay], edgecolor="black", linewidth=0.3)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.0f}", ha="center", va="bottom", fontsize=5.5)
    ax.set_xticks(range(len(CASES)))
    ax.set_xticklabels([c.replace(", ", "\n") for _, c in CASES])
    ax.set_ylim(0, 108)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="x", visible=False)


# ---- Fig: baseline (two fixed layouts) ----
fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.2))
grouped(ax[0], LAY[:2], "npu_meas", "Utilization (%)", "(a) NPU sequential read (Ramulator)")
grouped(ax[1], LAY[:2], "pim_analytical", "Utilization (%)", "(b) PIM all-bank GEMV (analytical)")
ax[0].legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(1.1, 1.22))
save(fig, "baseline")

# ---- Fig: contribution (all four layouts) ----
fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.3))
grouped(ax[0], LAY, "npu_meas", "Utilization (%)", "(a) NPU sequential read (Ramulator)")
grouped(ax[1], LAY, "pim_analytical", "Utilization (%)", "(b) PIM all-bank GEMV (analytical)")
ax[0].legend(loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(1.1, 1.24))
save(fig, "contribution")

# ---- Fig: trade-off scatter ----
fig, ax = plt.subplots(figsize=(COL_W, 2.6))
MARK = {"x8_pn7": "o", "x8_pn9": "s", "x16_pn7": "^", "x16_pn9": "v"}
for key, _ in CASES:
    for lay in LAY:
        e = R[key][lay]
        ax.scatter(e["npu_meas"], e["pim_analytical"], marker=MARK[key], s=28,
                   color=COLOR[lay], edgecolor="black", linewidth=0.4, zorder=3)
h1 = [plt.Line2D([], [], ls="", marker="o", color=COLOR[l], markeredgecolor="black", label=LABEL[l]) for l in LAY]
h2 = [plt.Line2D([], [], ls="", marker=MARK[k], color="white", markeredgecolor="black", label=c) for k, c in CASES]
ax.legend(handles=h1 + h2, loc="center left", frameon=False, fontsize=6, ncol=1, bbox_to_anchor=(0.02, 0.55))
ax.set_xlabel("NPU utilization, Ramulator (%)")
ax.set_ylabel("PIM utilization, analytical (%)")
ax.set_xlim(58, 100)
ax.set_ylim(-3, 103)
ax.annotate("ideal", xy=(98.5, 100), fontsize=6, ha="right", color="gray")
save(fig, "tradeoff")

# ---- Fig: LLaMA3-8B page-weighted summary ----
W = {(r["org"], r["layout"]): r for r in SHAPE_ROWS if r["tensor"].startswith("ALL")}
fig, ax = plt.subplots(1, 2, figsize=(FULL_W, 2.2), sharey=True)
for a, org in zip(ax, ("x8", "x16")):
    xs = range(len(LAY))
    npu = [float(W[org, l]["npu_util_meas"]) for l in LAY]
    pim = [float(W[org, l]["pim_util_analytical"]) for l in LAY]
    b1 = a.bar([x - 0.19 for x in xs], npu, 0.38, color="#3b6fb6", edgecolor="black", linewidth=0.3, label="NPU (Ramulator)")
    b2 = a.bar([x + 0.19 for x in xs], pim, 0.38, color="#e08a2c", edgecolor="black", linewidth=0.3, hatch="///", label="PIM (analytical)")
    for b, v in list(zip(b1, npu)) + list(zip(b2, pim)):
        a.text(b.get_x() + b.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=5.5)
    a.set_xticks(list(xs))
    a.set_xticklabels([LABEL[l].replace(" ", "\n", 1).replace("-", "-\n", 1) for l in LAY])
    a.set_ylim(0, 108)
    a.set_title(f"DDR4 {org}, weighted over 7 matrices")
    a.grid(axis="x", visible=False)
ax[0].set_ylabel("Utilization (%)")
ax[0].legend(loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(1.05, 1.25))
save(fig, "llama_weighted")

# ---- Fig: bank-group interleaving effect (probe runs, values from probe2.py / probe_x16.py output) ----
PROBE = [("x8", "column bits lowest", 65.83), ("x8", "bank-group bits lowest", 95.23),
         ("x8", "bank-group + bank lowest", 95.03), ("x16", "column bits lowest", 65.46),
         ("x16", "bank-group bit lowest", 94.82)]
fig, ax = plt.subplots(figsize=(COL_W, 1.7))
ys = list(range(len(PROBE)))[::-1]
cols = ["#e08a2c" if "column" in p[1] else "#3b6fb6" for p in PROBE]
ax.barh(ys, [p[2] for p in PROBE], color=cols, edgecolor="black", linewidth=0.3)
for y, p in zip(ys, PROBE):
    ax.text(p[2] + 0.8, y, f"{p[2]:.2f}", va="center", fontsize=6)
ax.axvline(100 * 4 / 6, ls="--", lw=0.7, color="gray")
ax.text(100 * 4 / 6, len(PROBE) - 0.35, r"4/6 = tCCD$_S$/tCCD$_L$", fontsize=6, color="gray", ha="center")
ax.set_yticks(ys)
ax.set_yticklabels([f"{p[0]}: {p[1]}" for p in PROBE])
ax.set_xlim(0, 108)
ax.set_xlabel("NPU sequential-read utilization (%)")
ax.grid(axis="y", visible=False)
save(fig, "bg_effect")

# ---- Fig: read-stream proxy vs analytical PIM model ----
fig, ax = plt.subplots(figsize=(COL_W, 2.3))
for key, _ in CASES:
    for lay, e in R[key].items():
        ax.scatter(e["pim_analytical"], e["pim_proxy"], marker=MARK[key], s=24,
                   color=COLOR.get(lay, "gray"), edgecolor="black", linewidth=0.4, zorder=3)
ax.plot([0, 100], [0, 100], ls="--", lw=0.7, color="gray")
ax.text(60, 52, "y = x", fontsize=6, color="gray", rotation=33)
ax.set_xlabel("PIM utilization, analytical all-bank model (%)")
ax.set_ylabel("PIM utilization, read-stream proxy\nin Ramulator (%)")
ax.set_xlim(-3, 103)
ax.set_ylim(-3, 103)
ax.legend(handles=h1 + h2, loc="lower right", frameon=False, fontsize=5.5, ncol=2)
save(fig, "proxy_vs_model")

# ---- LaTeX tables ----
def tex_layout(order):
    tag = {"col": "Co", "bg": "BG", "bank": "Ba", "row": "Ro"}
    return r"\texttt{" + " ".join(f"{tag[l]}{n}" for l, n in reversed(order)) + "}"


def write(name, s):
    open(os.path.join(TAB, name), "w").write(s)


rows = []
for key, cap in CASES:
    for i, lay in enumerate(LAY):
        e = R[key][lay]
        first = r"\multirow{4}{*}{" + cap.replace("$d_{in}$", "$d_{\\mathrm{in}}$") + "}" if i == 0 else ""
        rows.append(f"{first} & {LABEL[lay]} & {e['npu_meas']:.2f} & {e['pim_analytical']:.2f} & {e['pim_proxy']:.2f} \\\\")
    rows.append(r"\midrule")
rows = rows[:-1]
write("results_table.tex", r"""\begin{table}[t]
\centering
\caption{Bandwidth utilization (\%) of every layout. NPU: Ramulator. PIM: analytical all-bank model; the read-stream proxy is shown for reference.}
\label{tab:results}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrrr}
\toprule
Case & Layout & NPU & PIM & PIM proxy \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

rows = []
for org in ("x8", "x16"):
    for lay in LAY:
        w = W[org, lay]
        rows.append(f"{org} & {LABEL[lay]} & {float(w['npu_util_meas']):.2f} & {float(w['pim_util_analytical']):.2f} \\\\")
    rows.append(r"\midrule")
rows = rows[:-1]
write("weighted_table.tex", r"""\begin{table}[t]
\centering
\caption{Utilization (\%) over all seven LLaMA3-8B weight matrices of a decoder layer, weighted by their size in 2\,MB pages of weight data (152 pages with $d_{\mathrm{in}}{=}4096$, 56 for \texttt{down\_proj}, whose padded layout would occupy 64).}
\label{tab:weighted}
\begin{tabular}{llrr}
\toprule
DRAM & Layout & NPU & PIM \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

rows = []
for key, cap in CASES[:2]:
    for lay in ["NPU-friendly", "PIM-friendly", "PFM a=0.25", "PFM a=1.0"]:
        rows.append(f"{cap.replace('$d_{in}$', '$d_{\\mathrm{in}}$')} & {LABEL[lay]} & {tex_layout(R[key][lay]['order'])} \\\\")
write("layouts_table.tex", r"""\begin{table}[t]
\centering
\caption{Layouts of the 15 line-index bits of a 2\,MB page (DDR4 x8), MSB on the left. Co: column, BG: bank group, Ba: bank, Ro: row; the number is the bit count. $\alpha{\ge}1$ gives the same layout as $\alpha{=}1$.}
\label{tab:layouts}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{lll}
\toprule
Case & Layout & Bits 14 $\rightarrow$ 0 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")

# per-tensor table (x8)
SH = {"q_proj": "4096$\\times$4096", "k_proj": "4096$\\times$1024", "v_proj": "4096$\\times$1024",
      "o_proj": "4096$\\times$4096", "gate_proj": "4096$\\times$14336", "up_proj": "4096$\\times$14336",
      "down_proj": "14336$\\times$4096"}
rows = []
for t in SH:
    cells = []
    for lay in LAY:
        r = next(r for r in SHAPE_ROWS if r["org"] == "x8" and r["tensor"] == t and r["layout"] == lay)
        cells.append(f"{float(r['npu_util_meas']):.1f}/{float(r['pim_util_analytical']):.1f}")
    pages = next(r for r in SHAPE_ROWS if r["tensor"] == t)["pages"]
    rows.append(f"\\texttt{{{t.replace('_', chr(92) + '_')}}} & {SH[t]} & {pages} & " + " & ".join(cells) + r" \\")
write("shapes_table.tex", r"""\begin{table*}[t]
\centering
\caption{Per-matrix utilization on DDR4 x8, shown as NPU\,\% / PIM\,\%. Shapes are $d_{\mathrm{in}}\times d_{\mathrm{out}}$ of $y = xW$ in FP16. Pages = size of the weight data in 2\,MB pages.}
\label{tab:shapes}
\begin{tabular}{llrcccc}
\toprule
Matrix & Shape & Pages & NPU-friendly & PIM-friendly & PFM $\alpha$=0.25 & PFM $\alpha$=2 \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table*}
""")

# numbers used in text
x8 = {l: W["x8", l] for l in LAY}
num = {
    "npuNF": float(x8["NPU-friendly"]["npu_util_meas"]), "pimNF": float(x8["NPU-friendly"]["pim_util_analytical"]),
    "npuPF": float(x8["PIM-friendly"]["npu_util_meas"]), "pimPF": float(x8["PIM-friendly"]["pim_util_analytical"]),
    "npuLo": float(x8["PFM a=0.25"]["npu_util_meas"]), "pimLo": float(x8["PFM a=0.25"]["pim_util_analytical"]),
    "npuHi": float(x8["PFM a=2.0"]["npu_util_meas"]), "pimHi": float(x8["PFM a=2.0"]["pim_util_analytical"]),
}
print("page-weighted x8:", {k: round(v, 2) for k, v in num.items()})
print("row conflicts, x8 pn9 PIM proxy:",
      {n: MEAS_X8["9"][n]["pim"]["row_conflicts"] for n in MEAS_X8["9"]})
print("wrote figures to", FIG, "and tables to", TAB)

# ---- TikZ: bit-layout diagram (x8, both d_in classes) ----
FILL = {"col": "colCo", "bg": "colBG", "bank": "colBa", "row": "colRo"}
TAG = {"col": "Co", "bg": "BG", "bank": "Ba", "row": "Ro"}


def bits(order):
    out = []
    for l, n in order:
        out += [l] * n
    return out  # index = bit position (LSB first)


lines = [r"\begin{tikzpicture}[x=0.42cm,y=0.42cm,font=\scriptsize]"]
y = 0
for key, cap, pn in (("x8_pn7", "$d_{\\mathrm{in}}{=}4096$", 7), ("x8_pn9", "$d_{\\mathrm{in}}{=}14336$", 9)):
    lines.append(rf"\node[anchor=west,font=\scriptsize\bfseries] at (-0.1,{y + 0.6}) {{{cap}: PIM unit window = bits {pn}..{pn + 3}}};")
    for lay in ["NPU-friendly", "PIM-friendly", "PFM a=0.25", "PFM a=1.0"]:
        b = bits(R[key][lay]["order"])
        lab = LABEL[lay]
        lines.append(rf"\node[anchor=east] at (-0.2,{y - 0.5}) {{{lab}}};")
        for pos in range(15):
            x = 14 - pos
            lines.append(rf"\draw[fill={FILL[b[pos]]}] ({x},{y - 1}) rectangle ++(1,1) node[pos=.5] {{{TAG[b[pos]][0:2]}}};")
        lines.append(rf"\draw[very thick,red!70!black] ({14 - (pn + 3)},{y - 1}) rectangle ({15 - pn},{y});")
        y -= 1.15
    y -= 1.6
for pos in range(15):
    lines.append(rf"\node at ({14 - pos + 0.5},{y + 0.6}) {{\tiny {pos}}};")
lines.append(rf"\node[anchor=east] at (-0.2,{y + 0.6}) {{\tiny line-index bit}};")
lines.append(r"\end{tikzpicture}")
open(os.path.join(FIG, "layouts_tikz.tex"), "w").write("\n".join(lines) + "\n")
print("wrote layouts_tikz.tex")

# ---- region mapper table ----
RM = json.load(open(os.path.join(RES, "region_mapper.json")))
rows = []
for plan, e in RM["multi_region"].items():
    lab = plan.replace("a=0.25", r"$\alpha$=0.25").replace("a=2.0", r"$\alpha$=2")
    rows.append(f"{lab} & {len(e['layouts'])} & {''.join(map(str, e['region_layout']))} & {e['util']:.2f} \\\\")
n_ok = sum(v["identical"] for k in RM["equivalence"].values() for v in k.values())
n_all = sum(len(k) for k in RM["equivalence"].values())
write("region_table.tex", r"""\begin{table}[t]
\centering
\caption{One NPU stream over eight 2\,MB regions (six $d_{\mathrm{in}}{=}4096$ pages, two \texttt{down\_proj} pages) with \texttt{PFMRegionMapper} inside Ramulator (DDR4 x8). The region table gives the layout id of each region.}
\label{tab:region}
\setlength{\tabcolsep}{4pt}
\begin{tabular}{lccr}
\toprule
Plan & Layouts & Region table & NPU \% \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}
""")
open(os.path.join(TAB, "region_numbers.tex"), "w").write(
    f"\\newcommand{{\\eqok}}{{{n_ok}}}\n\\newcommand{{\\eqall}}{{{n_all}}}\n")
print("region mapper: identical", n_ok, "of", n_all)
