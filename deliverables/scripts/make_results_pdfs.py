#!/usr/bin/env python3
"""Build Baseline_Results.pdf and Contribution_Results.pdf from the result files."""
import json
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "..", "v2", "results")
OUT = os.path.join(ROOT, "results_pdfs")
os.makedirs(OUT, exist_ok=True)

R = json.load(open(os.path.join(RES, "final.json")))
RM = json.load(open(os.path.join(RES, "region_mapper.json")))
PAGES = {"pn7": 152, "pn9": 56}
CASES = [("x8_pn7", r"x8, $d_{\mathrm{in}}{=}4096$"), ("x8_pn9", r"x8, $d_{\mathrm{in}}{=}14336$"),
         ("x16_pn7", r"x16, $d_{\mathrm{in}}{=}4096$"), ("x16_pn9", r"x16, $d_{\mathrm{in}}{=}14336$")]
LAB = {"NPU-friendly": "NPU-friendly", "PIM-friendly": "PIM-friendly",
       "PFM a=0.25": r"PFM $\alpha$=0.25", "PFM a=2.0": r"PFM $\alpha$=2"}


def weighted(org, lay):
    a, b = R[f"{org}_pn7"][lay], R[f"{org}_pn9"][lay]
    t = PAGES["pn7"] + PAGES["pn9"]
    return ((PAGES["pn7"] * a["npu_meas"] + PAGES["pn9"] * b["npu_meas"]) / t,
            (PAGES["pn7"] * a["pim_analytical"] + PAGES["pn9"] * b["pim_analytical"]) / t)


PRE = r"""\documentclass[10pt,a4paper]{article}
\usepackage[margin=2cm]{geometry}
\usepackage{graphicx,booktabs,float,amsmath,array}
\usepackage[hidelinks]{hyperref}
\usepackage{times}
\setlength{\parindent}{0pt}\setlength{\parskip}{0.5em}
\graphicspath{{../figures/}}
\begin{document}
\begin{center}
{\Large\bfseries %s}\\[0.3em]
{\normalsize Dual-View Address Mapping for NPU--PIM Memory on Ramulator 2 --- Group 13}\\[0.2em]
{\small Yash Gupta, Nikunj Ladumor, Vamshi Kumar Kanduri, Priyanshu Ganjeer --- Dept.\ of CSE, IIT Guwahati}
\end{center}
"""
END = "\n\\end{document}\n"


def sgn(v):
    return f"{v:+.2f}"


# ------------------------------------------------------------------ baseline
rows = []
for key, cap in CASES:
    for lay in ("NPU-friendly", "PIM-friendly"):
        e = R[key][lay]
        rows.append(f"{cap} & {LAB[lay]} & {e['npu_meas']:.2f} & {e['pim_analytical']:.2f} \\\\")
    rows.append(r"\midrule")
rows = rows[:-1]
wrows = []
for org in ("x8", "x16"):
    for lay in ("NPU-friendly", "PIM-friendly"):
        n, p = weighted(org, lay)
        wrows.append(f"{org} & {LAB[lay]} & {n:.2f} & {p:.2f} \\\\")

baseline = PRE % "Baseline Results" + r"""
\textbf{Setup.} Two fixed address layouts for the 15 line-index bits of a 2\,MB page, listed from bit 0 upward:
\textbf{NPU-friendly} (bank group, column, bank, row) and \textbf{PIM-friendly} (column, bank group, bank, row).
NPU utilization is measured in Ramulator~2.1 (DDR4-2400R, one channel) for a sequential read stream
(utilization $= 4\times$reads$/$DRAM cycles). PIM utilization is an analytical all-bank GEMV model using Ramulator's DDR4 timings.
Weights are the LLaMA3-8B matrices: six with $d_{\mathrm{in}}{=}4096$ and \texttt{down\_proj} with $d_{\mathrm{in}}{=}14336$.

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{baseline.pdf}
\caption{Baseline bandwidth utilization. (a) NPU sequential read (Ramulator); (b) PIM all-bank GEMV (analytical).}
\end{figure}

\begin{table}[H]\centering
\caption{Baseline utilization (\%) per case.}
\begin{tabular}{llrr}
\toprule
Case & Layout & NPU & PIM \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering
\caption{Baseline utilization (\%) over all seven weight matrices of a decoder layer, weighted by their size in 2\,MB pages of weight data (152 pages with $d_{\mathrm{in}}{=}4096$, 56 for \texttt{down\_proj}).}
\begin{tabular}{llrr}
\toprule
DRAM & Layout & NPU & PIM \\
\midrule
""" + "\n".join(wrows) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=0.62\linewidth]{bg_effect.pdf}
\caption{Why the NPU-friendly layout is fast: with a bank-group bit lowest, consecutive reads alternate bank groups (4-cycle spacing, $t_{CCD\_S}$); with column bits lowest every read waits 6 cycles ($t_{CCD\_L}$), capping utilization near $4/6=66.7\%$.}
\end{figure}

\textbf{Observations.}
\begin{itemize}
\item Each fixed layout serves one device only: NPU-friendly gives about 95\% NPU but 24\% PIM (x8, $d_{\mathrm{in}}{=}4096$), because two of the four bits that select a PIM unit are column bits and four units share each bank. PIM-friendly gives 96\% PIM but only about 66\% NPU.
\item For \texttt{down\_proj} both layouts give 3.95\% PIM: its unit-selection bits (9--12) contain two row bits, so units share banks and switch rows on every access. Both baselines are the standard orders and are designed for $d_{\mathrm{in}}{=}4096$.
\end{itemize}
""" + END

# ------------------------------------------------------------------ contribution
rows = []
for key, cap in CASES:
    for lay in ("NPU-friendly", "PIM-friendly", "PFM a=0.25", "PFM a=2.0"):
        e = R[key][lay]
        rows.append(f"{cap} & {LAB[lay]} & {e['npu_meas']:.2f} & {e['pim_analytical']:.2f} & {e['pim_proxy']:.2f} \\\\")
    rows.append(r"\midrule")
rows = rows[:-1]

comp = []
for key, cap in CASES:
    for pfm, base in (("PFM a=0.25", "NPU-friendly"), ("PFM a=2.0", "PIM-friendly")):
        a, b = R[key][pfm], R[key][base]
        comp.append(f"{cap} & {LAB[pfm]} vs {LAB[base]} & {sgn(a['npu_meas'] - b['npu_meas'])} & {sgn(a['pim_analytical'] - b['pim_analytical'])} \\\\")
    comp.append(r"\midrule")
for org in ("x8", "x16"):
    for pfm, base in (("PFM a=0.25", "NPU-friendly"), ("PFM a=2.0", "PIM-friendly")):
        an, ap = weighted(org, pfm)
        bn, bp = weighted(org, base)
        comp.append(f"{org}, whole layer & {LAB[pfm]} vs {LAB[base]} & {sgn(an - bn)} & {sgn(ap - bp)} \\\\")
wrows = []
for org in ("x8", "x16"):
    for lay in ("NPU-friendly", "PIM-friendly", "PFM a=0.25", "PFM a=2.0"):
        n, p = weighted(org, lay)
        wrows.append(f"{org} & {LAB[lay]} & {n:.2f} & {p:.2f} \\\\")
    wrows.append(r"\midrule")
wrows = wrows[:-1]
mr = []
for plan, e in RM["multi_region"].items():
    mr.append(f"{plan.replace('a=0.25', '$\\alpha$=0.25').replace('a=2.0', '$\\alpha$=2')} & {len(e['layouts'])} & {e['util']:.2f} \\\\")
n_ok = sum(v["identical"] for k in RM["equivalence"].values() for v in k.values())
n_all = sum(len(k) for k in RM["equivalence"].values())
x8w = {l: weighted("x8", l) for l in LAB}

contribution = PRE % "Contribution Results" + r"""
\textbf{What is new over the baselines.} (1) A PuLP/CBC MILP chooses the layout of the 15 line-index bits to maximize $\mathrm{NPU}(f)+\alpha\,\mathrm{PIM}(f)$ (we show $\alpha{=}0.25$, NPU-leaning, and $\alpha{=}2$, PIM-leaning).
(2) \texttt{PFMRegionMapper}, a new Ramulator address mapper, gives every 2\,MB region its own layout.
NPU utilization is measured in Ramulator~2.1 (DDR4-2400R); PIM utilization is an analytical all-bank GEMV model.

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{contribution.pdf}
\caption{All layouts. The first two bars of each group are the baselines; green and purple are the optimized (PFM) layouts.}
\end{figure}

\begin{table}[H]\centering
\caption{Utilization (\%) of every layout. The read-stream proxy (PIM order replayed as ordinary reads in Ramulator) is shown for reference.}
\begin{tabular}{llrrr}
\toprule
Case & Layout & NPU & PIM & PIM proxy \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{table}[H]\centering
\caption{Comparison over the baseline: change in utilization (percentage points) of each optimized layout against the baseline it is closest to.}
\begin{tabular}{llrr}
\toprule
Case & Comparison & $\Delta$NPU & $\Delta$PIM \\
\midrule
""" + "\n".join(comp) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=\linewidth]{llama_weighted.pdf}
\caption{Whole LLaMA3-8B decoder layer: all seven weight matrices, weighted by their size in 2\,MB pages of weight data.}
\end{figure}

\begin{table}[H]\centering
\caption{Whole decoder layer, utilization (\%).}
\begin{tabular}{llrr}
\toprule
DRAM & Layout & NPU & PIM \\
\midrule
""" + "\n".join(wrows) + r"""
\bottomrule
\end{tabular}
\end{table}

\begin{figure}[H]\centering
\includegraphics[width=0.55\linewidth]{tradeoff.pdf}
\caption{NPU vs.\ PIM utilization of every layout.}
\end{figure}

\begin{table}[H]\centering
\caption{\texttt{PFMRegionMapper} inside Ramulator: one NPU read stream over eight 2\,MB regions (six $d_{\mathrm{in}}{=}4096$ pages, two \texttt{down\_proj} pages), DDR4 x8. """ + f"For {n_ok} of {n_all}" + r""" layouts the mapper gives exactly the same cycle count as an independent Python implementation.}
\begin{tabular}{lrr}
\toprule
Plan & Layouts used & NPU \% \\
\midrule
""" + "\n".join(mr) + r"""
\bottomrule
\end{tabular}
\end{table}

\textbf{Analysis and comparison over the baseline.}
\begin{itemize}
""" + rf"""\item \textbf{{Whole layer (x8).}} PFM $\alpha$=0.25 keeps {x8w['PFM a=0.25'][0]:.1f}\% NPU utilization against {x8w['NPU-friendly'][0]:.1f}\% for NPU-friendly ({x8w['PFM a=0.25'][0]-x8w['NPU-friendly'][0]:+.1f} points) while raising PIM utilization from {x8w['NPU-friendly'][1]:.1f}\% to {x8w['PFM a=0.25'][1]:.1f}\% ({x8w['PFM a=0.25'][1]-x8w['NPU-friendly'][1]:+.1f} points). PFM $\alpha$=2 raises PIM from {x8w['PIM-friendly'][1]:.1f}\% (PIM-friendly) to {x8w['PFM a=2.0'][1]:.1f}\% ({x8w['PFM a=2.0'][1]-x8w['PIM-friendly'][1]:+.1f} points) for {x8w['PFM a=2.0'][0]-x8w['PIM-friendly'][0]:+.1f} NPU points.
""" + r"""\item \textbf{$d_{\mathrm{in}}{=}4096$ (six matrices).} PFM $\alpha$=0.25 keeps a bank-group bit at bit 0 for the NPU and places the other three bank-level bits in the PIM unit-selection bits: NPU stays at 95.2\% while PIM doubles from 24.0\% to 48.1\%. For $\alpha\ge1$ the optimizer selects a layout equivalent to PIM-friendly.
\item \textbf{\texttt{down\_proj}.} Both baselines give 3.95\% PIM. The optimizer moves the row bits out of the unit-selection bits: 48.1\% PIM at 90.2\% NPU ($\alpha$=0.25) or 96.1\% PIM ($\alpha$=2). Part of this gain comes from adapting the layout to the column length, which the fixed baselines (designed for $d_{\mathrm{in}}{=}4096$) do not do; the optimizer finds the shape-specific layout automatically.
\item \textbf{x16} (two bank groups) shows the same pattern; on $d_{\mathrm{in}}{=}4096$ the NPU-friendly layout already reaches 48\% PIM there, so $\alpha$=0.25 matches it rather than improving it.
\item \textbf{Trade-off.} On single-channel DDR4 no layout is best for both devices: x8 has only four bank-level bits, the NPU needs one at bit 0 and PIM needs all four in its unit-selection bits, which caps PIM at 50\% while the NPU runs at full speed. $\alpha$ chooses the point on this trade-off (switch at $\alpha{=}2/3$ in our model).
\item \textbf{Per-region mapping.} With a different layout in different regions of the same run, the $\alpha$=0.25 plan reaches 93.7\% NPU, matching the single-region results (93.9\% expected), so per-region layouts cost almost nothing.
\item \textbf{Caveats.} PIM utilization is analytical, not cycle-simulated; the memory is single-channel DDR4 rather than the paper's HBM2e-PIM, so the values do not reproduce the paper's Table III, only its trends.
\end{itemize}
""" + END

for name, body in (("Baseline_Results", baseline), ("Contribution_Results", contribution)):
    tex = os.path.join(OUT, name + ".tex")
    open(tex, "w").write(body)
    r = subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-halt-on-error", name + ".tex"],
                       cwd=OUT, capture_output=True, text=True)
    if r.returncode:
        log = open(os.path.join(OUT, name + ".log"), errors="replace").read()
        raise SystemExit("\n".join(l for l in log.splitlines() if l.startswith("!"))[:800])
    subprocess.run(["latexmk", "-c", "-quiet", name + ".tex"], cwd=OUT, capture_output=True)
    print("built", name + ".pdf")
