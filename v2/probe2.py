from rmlib import remap, check_bijection, run_trace
N_BITS = 15                      # 2 MB superpage / 64 B line = 32768 lines
LAYOUTS = {
  "A_col_first":      [("col",7),("bg",2),("bank",2),("row",4)],
  "B_bg_first":       [("bg",2),("col",7),("bank",2),("row",4)],
  "C_bg_bank_first":  [("bg",2),("bank",2),("col",7),("row",4)],
}
print(f"{'layout':<18}{'util%':>8}{'cycles':>9}{'rowhit':>8}{'rowmiss':>9}{'conf':>6}")
for name, order in LAYOUTS.items():
    check_bijection(order, N_BITS)
    addrs = [remap(i, order) for i in range(1 << N_BITS)]   # remap() returns byte addresses
    r = run_trace(addrs, name)
    print(f"{name:<18}{r['util']:8.2f}{r['cycles']:9d}{r['row_hits']:8d}{r['row_misses']:9d}{r['row_conflicts']:6d}")
