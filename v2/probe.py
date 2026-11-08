import os, sys, json, pprint
sys.path.insert(0, os.environ.get("RAMULATOR_PY", os.path.expanduser("~/pfm_assignment/ramulator2/python")))
import ramulator

N = 8192
with open("traces/probe_seq.trace", "w") as f:
    for i in range(N):
        f.write(f"0 {i*64}\n")          # raw sequential 64 B reads, no remap

frontend = ramulator.frontend.SimpleO3(
    clock_ratio=8, traces=["traces/probe_seq.trace"], num_expected_insts=N,
    translation=ramulator.translation.NoTranslation(max_addr=2147483648))
ddr4 = ramulator.dram.DDR4(org_preset="DDR4_8Gb_x8", timing_preset="DDR4_2400R",
                           rank=1, verbose=False)
ctrl = ramulator.controller.GenericDDR(
    dram=ddr4, scheduler=ramulator.scheduler.FRFCFS(),
    refresh_manager=ramulator.refresh_manager.AllBank(),
    row_policy=ramulator.row_policy.Open(),
    addr_mapper=ramulator.addr_mapper.RoBaRaCoCh())
mem = ramulator.memory_system.GenericDRAM(
    clock_ratio=3, controllers=[ctrl],
    channel_mapper=ramulator.channel_mapper.CacheLineInterleave())
sim = ramulator.Simulation(frontend, mem)
sim.run()
pprint.pprint(json.loads(json.dumps(sim.stats, default=str)), width=110, compact=True)
