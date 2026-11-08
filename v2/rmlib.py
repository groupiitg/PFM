import os, sys, json
sys.path.insert(0, os.environ.get("RAMULATOR_PY", os.path.expanduser("~/pfm_assignment/ramulator2/python")))
import ramulator

PEAK_MBPS = 19200.0                      # DDR4-2400 x64
FIELD_LSB = {"col": 6, "bg": 13, "bank": 15, "row": 17}   # RoBaRaCoCh field positions (x8), checked by probe2.py
FIELD_CAP = {"col": 7, "bg": 2, "bank": 2, "row": 16}

def remap(line, order):
    """Pure bit permutation: 'order' lists (field, nbits) from LSB of line index."""
    used = dict.fromkeys(FIELD_LSB, 0); ha = 0
    for name, n in order:
        v = line & ((1 << n) - 1); line >>= n
        ha |= v << (FIELD_LSB[name] + used[name]); used[name] += n
        assert used[name] <= FIELD_CAP[name], f"too many bits for {name}"
    assert line == 0, "line index has leftover bits"
    return ha

def check_bijection(order, nbits):
    s = {remap(i, order) for i in range(1 << nbits)}
    assert len(s) == 1 << nbits, "remap is not one-to-one"

def run_trace(addrs, tag, org="DDR4_8Gb_x8"):
    assert len(set(addrs)) == len(addrs), "duplicate addresses in trace"
    assert all(a % 64 == 0 for a in addrs)
    os.makedirs("traces", exist_ok=True)
    path = f"traces/{tag}.trace"
    with open(path, "w") as f:
        for a in addrs: f.write(f"0 {a}\n")
    n = len(addrs)
    fe = ramulator.frontend.SimpleO3(
        clock_ratio=8, traces=[path], num_expected_insts=n,
        translation=ramulator.translation.NoTranslation(max_addr=2147483648))
    ddr4 = ramulator.dram.DDR4(org_preset=org, timing_preset="DDR4_2400R",
                               rank=1, verbose=False)
    ctrl = ramulator.controller.GenericDDR(
        dram=ddr4, scheduler=ramulator.scheduler.FRFCFS(),
        refresh_manager=ramulator.refresh_manager.AllBank(),
        row_policy=ramulator.row_policy.Open(),
        addr_mapper=ramulator.addr_mapper.RoBaRaCoCh())
    mem = ramulator.memory_system.GenericDRAM(
        clock_ratio=3, controllers=[ctrl],
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
    sim = ramulator.Simulation(fe, mem); sim.run()
    c = sim.stats["memory_system"]["controller"]
    assert c["num_read_reqs_served"] == n and c["num_read_reqs_forwarded"] == 0, \
        f"only {c['num_read_reqs_served']}/{n} reads reached DRAM"
    return {"tag": tag, "reads": n, "cycles": c["cycles"],
            "util": 100.0 * n * 4 / c["cycles"],
            "mbps": c["read_throughput_MBps"],
            "row_hits": c["row_hits"], "row_misses": c["row_misses"],
            "row_conflicts": c["row_conflicts"], "lat": c["avg_read_latency"]}
