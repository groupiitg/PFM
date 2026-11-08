# Dual-view address mapping for NPU-PIM memory (PFM) on Ramulator 2.0

Course project, Group 13. This is a reproduction of the address-mapping optimizer from:

S. Zhao et al., "Rethinking Unified Memory for NPU-PIM Systems: Dual-View Memory for Dynamic Inference of LLM", arXiv:2608.06989.

## What it does

An NPU and a PIM unit share one DRAM but want different layouts for the same weight matrix. The project does three things:

- A PuLP MILP assigns the 15 line-index bits of a 2 MB page to column, bank-group, bank and row bits of a DDR4 device. It maximizes NPU(f) + alpha * PIM(f).
- A new Ramulator 2.0 address mapper, `PFMRegionMapper`, gives every 2 MB memory region its own layout (a table of layouts plus a region-to-layout table). Its results are checked cycle for cycle against an independent Python bit permutation.
- Three kinds of layout are compared on the LLaMA3-8B weight shapes: NPU-friendly, PIM-friendly and jointly optimized (PFM, alpha = 0.25 / 1 / 2). NPU sequential-read utilization is simulated in Ramulator. PIM GEMV utilization is computed with an analytical all-bank model that uses Ramulator's DDR4-2400R timings.

Main result (DDR4 x8, page-weighted over the seven weight matrices of a decoder layer):

| Layout | NPU % (Ramulator) | PIM % (analytical) |
|---|---|---|
| NPU-friendly | 95.23 | 18.62 |
| PIM-friendly | 65.83 | 71.30 |
| PFM alpha=0.25 | 93.82 | 48.06 |
| PFM alpha=2 | 64.71 | 96.12 |

## Layout

```
ramulator_patch/        the Ramulator 2.0 change (PFMRegionMapper) as a patch + the C++ file
v2/                     experiment code and raw results
  rmlib.py              one-to-one remapper + Ramulator runner (asserts all reads reach DRAM)
  probe.py, probe2.py,  checks of the metric and of the DDR4 field positions
  probe_x16.py
  pfm_opt2.py           first MILP (bit budget, NPU score, layouts, alpha sweep)
  pim_model.py/pimlib.py analytical all-bank PIM model
  pfm_opt3.py           final MILP with the unit-window PIM score -> results/opt3.json
  run_eval*.py          earlier measurement runs
  run_final.py          final measurement of all layouts -> results/final.json
  make_report_data.py   page-weighted table -> results/shape_table.csv
  run_region_mapper.py  all layouts through PFMRegionMapper + multi-region run -> results/region_mapper.json
  results/              JSON/CSV results
deliverables/
  scripts/make_figures.py        figures (PDF/PNG) and LaTeX tables from v2/results
  scripts/package_submission.py  builds deliverables/submission/
  report/               IEEE conference report (LaTeX)
  slides/               presentation (Beamer)
```

## Reproduce

Requirements: Python 3.10+, PuLP, matplotlib, g++ 12+ (C++20), CMake, and TeX Live for the report.

Build Ramulator 2.0 (commit c6c7a5d) with the PFM mapper:

```bash
git clone https://github.com/CMU-SAFARI/ramulator2.git
cd ramulator2 && git checkout c6c7a5d
git apply ../ramulator_patch/pfm_region_mapper.patch
./build.sh -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cd ..
export RAMULATOR_PY=$PWD/ramulator2/python
```

Then run the experiments:

```bash
cd v2
python3 pfm_opt2.py                # results/mappings_x8.json
python3 pfm_opt3.py                # results/opt3.json (needs Ramulator importable)
python3 run_final.py               # results/final.json (runs Ramulator)
python3 make_report_data.py        # results/shape_table.csv
python3 run_region_mapper.py       # results/region_mapper.json
cd ../deliverables
python3 scripts/make_figures.py
(cd report && latexmk -pdf main.tex)
(cd slides && latexmk -pdf slides.tex)
python3 scripts/package_submission.py --repo-url <URL>
```

## Limitations

- PIM utilization is analytical, not cycle-simulated.
- The memory is single-channel DDR4, not HBM2e-PIM.
- The fixed baselines are designed for d_in = 4096, so they are weak on down_proj.

The report discusses all three.
