import json
import rmlib
from collections import defaultdict
import pfm_opt2 as M

CCDL, RCD, RP = 6, 16, 16
NB = 15
GEOM = {  # field LSBs, bank-group bits, units, caps
  "x8":  dict(lsb={"col":6,"bg":13,"bank":15,"row":17}, bgbits=2, U=16, cap=M.ORGS["x8"]),
  "x16": dict(lsb={"col":6,"bg":13,"bank":14,"row":16}, bgbits=1, U=8,  cap=M.ORGS["x16"]),
}

def pim_util(order, pn, g):
    rmlib.FIELD_LSB.update(g["lsb"])
    rmlib.FIELD_CAP.update({"col":7,"bg":g["bgbits"],"bank":2,"row":16})
    U, T, ncol = g["U"], 1 << pn, 1 << (NB - pn)
    L = g["lsb"]; bgm = (1 << g["bgbits"]) - 1
    openrow, total, steps = {}, 0, 0
    for grp in range(ncol // U):
        for t in range(T):
            cost = defaultdict(int)
            for u in range(U):
                ha = rmlib.remap((U*grp + u) * T + t, order)
                bank = ((ha >> L["bg"]) & bgm, (ha >> L["bank"]) & 3)
                row = ha >> L["row"]
                c = CCDL
                if bank not in openrow: c += RCD
                elif openrow[bank] != row: c += RP + RCD
                openrow[bank] = row
                cost[bank] += c
            total += max(cost.values()); steps += 1
    return 100.0 * steps * CCDL / total

