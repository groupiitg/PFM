import rmlib
rmlib.FIELD_LSB.update({"col": 6, "bg": 13, "bank": 14, "row": 16})
rmlib.FIELD_CAP.update({"col": 7, "bg": 1, "bank": 2, "row": 16})
from rmlib import remap, check_bijection, run_trace
NB = 15
L = {"A_col_first": [("col",7),("bg",1),("bank",2),("row",5)],
     "B_bg_first":  [("bg",1),("col",7),("bank",2),("row",5)]}
print(f"{'layout':<14}{'util%':>8}{'rowhit':>8}{'rowmiss':>9}{'conf':>6}")
for n, o in L.items():
    check_bijection(o, NB)
    r = run_trace([remap(i, o) for i in range(1 << NB)], "x16_" + n, org="DDR4_8Gb_x16")
    print(f"{n:<14}{r['util']:8.2f}{r['row_hits']:8d}{r['row_misses']:9d}{r['row_conflicts']:6d}")
