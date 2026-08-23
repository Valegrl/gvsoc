# Benchmark utilities

Two scripts turn a GVSoC run of a generated pipeline app into a performance
report:

| script | in | out |
| --- | --- | --- |
| `parse_bench.py` | run log + the app `.c` that produced it | per-region report, CSV, per-stage figures |
| `latency_throughput_bars.py` | one or more of those CSVs | latency / throughput bar charts |

## Getting a log

Each region of interest is delimited in the app by
`mempool_start_benchmark()` / `mempool_stop_benchmark()`, which write the mempool
trace CSR `0x7d0`.  With `soft_hier/gvsoc_core.patch` applied, every closed region
prints one line per core:

```
[BENCH] cycles=4123 ns=4123 start=10500 end=14623 region=7 hart=3 path=/chip/cluster_2/...
```

So the log is simply the run's stdout:

```sh
./install/bin/gvsoc --target=pulp.chips.flex_cluster.flex_cluster \
    --binary cevit_ctt_TensorPool run | tee cevit_ctt_TensorPool.log
```

A record names its region only by INDEX.  The kernel names, GEMM shapes and
transfer sizes come from the app source, which is why `--app` is worth passing.

## `parse_bench.py`

```sh
./parse_bench.py LOG --app APP.c [--csv OUT.csv] [--plot_ctt] [--plot_cwt]
```

It rebuilds each cluster's region table from the app source, matches it against
the log, and prints:

* one block per cluster — every region with its span, unit (TE / PE / LSU /
  DMA:LPDDR / DMA:NoC / DMA:L1), achieved rate and utilization against that
  unit's peak;
* utilization aggregated per kernel and per unit;
* the share of total cycles per kernel class;
* the pipeline: per-cluster busy window, critical path, end-to-end cycles.

With `--plot_ctt` / `--plot_cwt` it also prints the pipeline model (per-stage
compute vs transfer, bottleneck `T_pipe`, steady-state latency) and writes a
two-panel figure — compute vs transfer on top, the same bars subdivided by kernel
below — to `<flavour>_bench_plots/` next to the log.

Useful flags:

| flag | effect |
| --- | --- |
| `--app FILE` | the app `.c` the log came from; without it the run is reported raw, by region index |
| `--cevit` / `--celere` | pick the generator model; default is detected from the tables the app declares |
| `--partial` | label clusters that stopped part-way (a hung or still-running job); implied by `--celere` |
| `--raw` | never label, even with `--app` |
| `--csv FILE` | write the per-region table (input of the other script) |
| `--plot_ctt [PNG]`, `--plot_cwt [PNG]` | the stage figure for the serialized / overlapped schedule |
| `--ymax MS`, `--plot-label NAME` | fix the y axis / the figure title, to compare two runs |
| `--freq-hz F` | clock for cycles → ms (default 1e9, so 1 cycle = 1 ns) |
| `--dram-peak BPC` | off-chip peak per channel; default is read from the DRAMSys `MAX BW` footer |

Notes:

* both app models track the CURRENT generators — an app from an older
  `cevit_pipeline.py` / `celere_pipeline.py` reports raw rather than being
  labelled from a table it no longer matches;
* a region count that disagrees with the app drops that cluster to raw ("is this
  log from this app?"), which usually means log and `.c` are from different runs;
* rows flagged `<- over link peak` measure when the DMA engine was released, not
  when the data landed; everything derived from them is a lower bound.

## `latency_throughput_bars.py`

```sh
./latency_throughput_bars.py cevit_*_TensorPool_*.csv [--out lt.png]
```

Reads the `--csv` tables above, one config per file, and rebuilds each config's
bottleneck period exactly as `parse_bench.py` does (lane = sum of its rows,
stage = slowest lane, `T_pipe` = slowest stage), then plots

```
latency    = steps * T_pipe   [ms]           steps = n_stages (ctt), 2 * n_stages (cwt)
throughput = 1 / T_pipe       [antennas/ms]
```

The schedule comes from the file name (`..._ctt_...` / `..._cwt_...`), or from
`--mode`.  `--ctt` / `--cwt` keep only one mode, `--freq-hz` sets the clock.  The
default output is `<flavour>_bench_plots/<TAG>_latency_throughput[_<file>].png`,
next to the input — the same directory `parse_bench.py` plots into.

## Example

```sh
./parse_bench.py cevit_ctt_TensorPool.log --app cevit_ctt_TensorPool.c --cevit \
    --csv cevit_ctt_TensorPool.csv --plot_ctt
./parse_bench.py cevit_cwt_TensorPool.log --app cevit_cwt_TensorPool.c --cevit \
    --csv cevit_cwt_TensorPool.csv --plot_cwt
./latency_throughput_bars.py cevit_ctt_TensorPool.csv cevit_cwt_TensorPool.csv
```

and the same for a celere run:

```sh
./parse_bench.py celere_ctt_TensorPool.log --app celere_ctt_TensorPool.c --celere \
    --csv celere_ctt_TensorPool.csv --plot_ctt
```

`--cevit` / `--celere` are optional — without one the flavour is detected from the
tables the app source declares, and the flag is only rejected if it disagrees with
what the file looks like.  Pass it anyway to make the intent explicit, and note
that `--celere` additionally implies `--partial`, so a run that hung or is still
going still reports (its clusters are labelled as far as they got).

The plots need `matplotlib`; without it the text report is still printed.

The other files here (`flex_libfp8.py`, `preload.py`, `trace_analyzer/`,
`trace_perfetto/`) are unrelated FlexCluster utilities.
