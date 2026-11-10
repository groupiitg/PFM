# Dual-view address mapping for NPU-PIM memory (PFM) on Ramulator 2

Course project, Group 13. This is a reproduction of the address-mapping optimizer from:

S. Zhao et al., "Rethinking Unified Memory for NPU-PIM Systems: Dual-View Memory for Dynamic Inference of LLM", arXiv:2608.06989.

## What it does

An NPU and a PIM unit share one DRAM but want different layouts for the same weight matrix. The project does three things:

- A PuLP MILP assigns the 15 line-index bits of a 2 MB page to column, bank-group, bank and row bits of a DDR4 device. It maximizes NPU(f) + alpha * PIM(f).
- A new Ramulator address mapper, `PFMRegionMapper` (Ramulator 2.1, the current release of Ramulator 2), gives every 2 MB memory region its own layout (a table of layouts plus a region-to-layout table). Its results are checked cycle for cycle against an independent Python bit permutation.
- Three kinds of layout are compared on the LLaMA3-8B weight shapes: NPU-friendly, PIM-friendly and jointly optimized (PFM, alpha = 0.25 / 1 / 2). NPU sequential-read utilization is simulated in Ramulator. PIM GEMV utilization is computed with an analytical all-bank model that uses Ramulator's DDR4-2400R timings.

Main result (DDR4 x8, over the seven weight matrices of a decoder layer, weighted by their size in 2 MB pages of weight data):

| Layout | NPU % (Ramulator) | PIM % (analytical) |
|---|---|---|
| NPU-friendly | 95.23 | 18.62 |
| PIM-friendly | 65.83 | 71.30 |
| PFM alpha=0.25 | 93.82 | 48.06 |
| PFM alpha=2 | 64.71 | 96.12 |

## Layout

```
ramulator_patch/        the Ramulator change (PFMRegionMapper) as a patch + the C++ file
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

Requirements: Linux (tested on Ubuntu under WSL2), Python 3.10+, g++ 12+ (C++20), CMake 3.14+, git, and TeX Live (`latexmk`, IEEEtran, beamer) for the report and slides.

All commands below are run from the top folder of this repository, in one terminal.

**1. Python environment** (PuLP is pinned to 2.9.0; PuLP 4.x changed its API and does not work with this code):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**2. Build Ramulator (version 2.1, commit c6c7a5d) with the PFM mapper** (the first build downloads its dependencies and takes a few minutes):

```bash
git clone https://github.com/CMU-SAFARI/ramulator2.git
cd ramulator2 && git checkout c6c7a5d
git apply ../ramulator_patch/pfm_region_mapper.patch
./build.sh -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cd ..
export RAMULATOR_PY=$PWD/ramulator2/python
```

**3. Run the experiments** (same terminal, so the virtual environment and `RAMULATOR_PY` are still set; about 10-15 minutes in total):

```bash
cd v2
python3 pfm_opt2.py                # optimizer, first model -> results/mappings_x8.json
python3 pfm_opt3.py                # optimizer, final model + PIM model -> results/opt3.json
python3 run_final.py               # all layouts in Ramulator -> results/final.json
python3 make_report_data.py        # page-weighted table -> results/shape_table.csv
python3 run_region_mapper.py       # PFMRegionMapper checks -> results/region_mapper.json
cd ..
```

**4. Figures, report, slides and submission folder:**

```bash
cd deliverables
python3 scripts/make_figures.py
cp figures/*.pdf figures/layouts_tikz.tex report/figures/ && cp tables/*.tex report/tables/
cp figures/*.pdf figures/layouts_tikz.tex slides/figures/
(cd report && latexmk -pdf main.tex)
(cd slides && latexmk -pdf slides.tex)
python3 scripts/package_submission.py --repo-url https://github.com/groupiitg/PFM
cd ..
```

The results in `v2/results/` are already in the repository, so step 4 can be run on its own (after step 1) without rebuilding Ramulator. If you open a new terminal later, run `source .venv/bin/activate` and `export RAMULATOR_PY=$PWD/ramulator2/python` again from the top folder.

## Limitations

- PIM utilization is analytical, not cycle-simulated.
- The memory is single-channel DDR4, not HBM2e-PIM.
- The fixed baselines are designed for d_in = 4096, so they are weak on down_proj.
- down_proj is weighted by its 56 pages of weight data; its padded layout would occupy 64 pages (at most 2.5 points difference in the weighted results).

The report discusses all of these.
