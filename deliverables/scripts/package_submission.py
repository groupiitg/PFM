#!/usr/bin/env python3
"""Collect the six submission items into deliverables/submission/."""
import argparse
import csv
import json
import os
import shutil
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "..", "v2", "results")
FIG = os.path.join(ROOT, "figures")
TAB = os.path.join(ROOT, "tables")
OUT = os.path.join(ROOT, "submission")


def zip_dir(src, dst, keep):
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for base, _, files in os.walk(src):
            for f in files:
                p = os.path.join(base, f)
                rel = os.path.relpath(p, src)
                if keep(rel):
                    z.write(p, rel)


def copy(names, dst):
    os.makedirs(dst, exist_ok=True)
    for src, new in names:
        shutil.copy(src, os.path.join(dst, new))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="<add the repository URL after pushing>")
    a = ap.parse_args()

    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    src_ext = (".tex", ".bib", ".pdf", ".cls")
    no_build = lambda rel: rel.endswith(src_ext) and not (rel.endswith("main.pdf") or rel.endswith("slides.pdf"))

    shutil.copy(os.path.join(ROOT, "report", "main.pdf"), os.path.join(OUT, "1_Report.pdf"))
    zip_dir(os.path.join(ROOT, "report"), os.path.join(OUT, "2_LaTeX_Source.zip"), no_build)
    shutil.copy(os.path.join(ROOT, "slides", "slides.pdf"), os.path.join(OUT, "3_Presentation.pdf"))
    zip_dir(os.path.join(ROOT, "slides"), os.path.join(OUT, "3_Presentation_source.zip"), no_build)

    final = json.load(open(os.path.join(RES, "final.json")))
    rows = [["case", "layout", "order_lsb_to_msb", "npu_util_ramulator", "pim_util_analytical", "pim_util_proxy"]]
    base_rows = [rows[0]]
    for key, lays in final.items():
        for name, e in lays.items():
            r = [key, name, " ".join(f"{l}{n}" for l, n in e["order"]),
                 round(e["npu_meas"], 2), round(e["pim_analytical"], 2), round(e["pim_proxy"], 2)]
            rows.append(r)
            if name in ("NPU-friendly", "PIM-friendly"):
                base_rows.append(r)

    b = os.path.join(OUT, "4_Baseline_Results")
    copy([(os.path.join(FIG, "baseline.pdf"), "baseline_utilization.pdf"),
          (os.path.join(FIG, "baseline.png"), "baseline_utilization.png"),
          (os.path.join(FIG, "bg_effect.pdf"), "bank_group_effect.pdf"),
          (os.path.join(FIG, "bg_effect.png"), "bank_group_effect.png")], b)
    csv.writer(open(os.path.join(b, "baseline_results.csv"), "w", newline="")).writerows(base_rows)
    open(os.path.join(b, "README.txt"), "w").write(
        "Baseline results: NPU-friendly and PIM-friendly layouts.\n"
        "baseline_utilization: NPU sequential-read utilization (Ramulator 2.1, DDR4-2400) and PIM GEMV\n"
        "utilization (analytical all-bank model) for DDR4 x8/x16 and d_in = 4096 / 14336.\n"
        "bank_group_effect: NPU utilization with column bits vs bank-group bits lowest.\n"
        "baseline_results.csv: the numbers behind the figure.\n")

    c = os.path.join(OUT, "5_Contribution_Results")
    copy([(os.path.join(FIG, f + ext), n + ext) for f, n in
          (("contribution", "joint_vs_baselines_utilization"), ("llama_weighted", "llama3_8b_decoder_layer"),
           ("tradeoff", "npu_pim_tradeoff"), ("proxy_vs_model", "pim_model_vs_proxy"))
          for ext in (".pdf", ".png")], c)
    copy([(os.path.join(TAB, t), t) for t in ("results_table.tex", "weighted_table.tex", "shapes_table.tex", "layouts_table.tex")]
         + [(os.path.join(TAB, "region_table.tex"), "region_mapper_table.tex"),
            (os.path.join(RES, "region_mapper.json"), "region_mapper_results.json"),
            (os.path.join(RES, "shape_table.csv"), "per_matrix_results.csv"),
            (os.path.join(RES, "opt3.json"), "optimized_mappings.json")], c)
    csv.writer(open(os.path.join(c, "all_layouts_results.csv"), "w", newline="")).writerows(rows)
    open(os.path.join(c, "README.txt"), "w").write(
        "Contribution results: layouts chosen by the PuLP MILP (PFM alpha = 0.25 / 1 / 2) against the baselines.\n"
        "joint_vs_baselines_utilization: all layouts, NPU (Ramulator) and PIM (analytical).\n"
        "llama3_8b_decoder_layer: page-weighted over the seven LLaMA3-8B weight matrices.\n"
        "npu_pim_tradeoff: NPU vs PIM utilization of every layout.\n"
        "pim_model_vs_proxy: analytical PIM model vs read-stream proxy in Ramulator.\n"
        "region_mapper_*: PFMRegionMapper inside Ramulator - cycle-identical check for all layouts and\n"
        "  one NPU run over 8 regions with a different layout per region.\n"
        "*.tex: LaTeX tables used in the report. *.csv / *.json: raw numbers and solved layouts.\n")

    open(os.path.join(OUT, "6_Source_Code_Link.txt"), "w").write(
        f"Source code: {a.repo_url}\n\nThe repository README explains setup and how to reproduce the results.\n")
    for base, _, files in os.walk(OUT):
        for f in sorted(files):
            print(os.path.relpath(os.path.join(base, f), OUT))


if __name__ == "__main__":
    main()
