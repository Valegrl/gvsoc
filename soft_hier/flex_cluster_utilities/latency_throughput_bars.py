#!/usr/bin/env python3
"""Latency and throughput of each measured pipeline configuration, as two bar charts.

Reads one or more per-kernel CSVs (the tables `parse_bench.py --csv` writes, one
config per file) and, per config:

    latency    = steps * T_pipe  [ms]        -- one execution end-to-end at regime
    throughput = 1 / T_pipe  [antennas/ms]   -- completions per ms at the bottleneck

T_pipe is rebuilt exactly as parse_bench.py reports it: rows are grouped by
(stage, cluster) and charged to compute or transfer by their `unit` column, a
lane's time is the SUM of its rows' span_cyc, a stage costs its slowest lane and
T_pipe is the slowest stage.  ctt walls a lane at compute + transfer, cwt at
din + max(compute, transfer - din), where din is the `stage input` read the
generated loop serializes in front of the compute.  Which of the two a file is
comes from its name (`..._ctt_...` / `..._cwt_...`), or from --mode.

Latency is NOT the critical path sum_s(compute_s + transfer_s): the generated
pipelines put a global barrier on every step, so every step costs the bottleneck
period T_pipe.  An item is resident one step per stage in ctt (latency =
n_stages * T_pipe) and two in cwt, whose double-buffered schedule offsets
consecutive stages by two steps (latency = 2 * n_stages * T_pipe).  So cwt
shortens the period and lengthens the flight: the modes separate in BOTH panels.

Usage
-----
    python latency_throughput_bars.py cevit_ctt_TensorPool_164.csv
    python latency_throughput_bars.py cevit_*_TensorPool_*.csv --out lt.png
    python latency_throughput_bars.py run.csv --mode cwt        # name does not say
"""

import argparse
import csv
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class PipelineRecord:
    pool:       str
    mode:       str
    t_pipe:     float   # bottleneck stage wall [ms] = steady-state period
    n_stages:   int
    n_clusters: int
    latency:    float   # end-to-end wall of one item [ms] = steps * T_pipe


# Units whose time crosses a link; everything else -- TE, PE, the DMA:L1 layout
# copies -- is a cluster's own compute.  Same split parse_bench.py plots.
TRANSFER_UNITS = ('DMA:NOC', 'DMA:LPDDR')
COMPUTE_UNITS = ('TE', 'PE', 'DMA:L1')

MODE_RE = re.compile(r'(?:^|_)(ctt|cwt)(?:_|$)', re.I)
FLAVOUR_RE = re.compile(r'^(cevit|celere)_', re.I)


def split_name(stem: str) -> Tuple[str, Optional[str]]:
    """-> (pool, mode) read out of a CSV file name.

    'cevit_ctt_TensorPool_164' -> ('TensorPool_164', 'ctt'); a name that does not
    say which mode it is returns None for it, and --mode has to.
    """
    m = MODE_RE.search(stem)
    mode = m.group(1).lower() if m else None
    pool = MODE_RE.sub('_', stem, count=1) if m else stem
    pool = FLAVOUR_RE.sub('', pool).strip('_')
    return pool or stem, mode


def flavour_of(stem: str) -> str:
    """-> 'cevit' / 'celere', naming the plot directory as parse_bench.py does."""
    m = FLAVOUR_RE.match(stem)
    return m.group(1).lower() if m else 'cevit'


def read_csv_record(path: Path, mode: Optional[str], freq_hz: float) -> PipelineRecord:
    """Rebuild one config's T_pipe and latency from a measured per-kernel CSV."""
    pool, name_mode = split_name(path.stem)
    mode = mode or name_mode
    if mode is None:
        raise SystemExit(f'{path}: cannot tell ctt from cwt by name; pass --mode')

    # (stage, cluster) -> [compute_cyc, transfer_cyc, stage_input_cyc]
    acc: Dict[tuple, List[float]] = {}
    unknown: set = set()
    with path.open(newline='') as f:
        for row in csv.DictReader(f):
            if not (row.get('cluster') or '').strip():
                continue
            span = (row.get('span_cyc') or '').strip()
            if not span:
                continue
            unit = (row.get('unit') or '').strip().upper()
            if unit.startswith(TRANSFER_UNITS):
                xfer = True
            elif unit.startswith(COMPUTE_UNITS):
                xfer = False
            else:
                unknown.add(unit)
                xfer = False        # keep the time visible rather than dropping it
            slot = acc.setdefault((int(float(row['stage'])), row['cluster'].strip()),
                                  [0.0, 0.0, 0.0])
            slot[1 if xfer else 0] += float(span)
            # the stage input is tracked apart: cwt cannot overlap it, while its
            # sibling `L1 -> LPDDR (stage output)` shares the unit but IS overlapped
            if xfer and 'stage input' in (row.get('region') or '').lower():
                slot[2] += float(span)
    if not acc:
        raise SystemExit(f'{path}: no usable rows (want the parse_bench.py --csv table)')
    if unknown:
        print(f'Warning: {path.name}: unrecognised unit(s) {sorted(unknown)} '
              'counted as compute')

    cycles_per_ms = freq_hz / 1e3
    stage_wall: Dict[int, float] = {}
    for (stage, _cluster), (comp, xfer_cyc, din) in acc.items():
        wall = (din + max(comp, xfer_cyc - din)) if mode == 'cwt' else comp + xfer_cyc
        # a stage's lanes run in parallel: it costs the slowest of them, not their sum
        stage_wall[stage] = max(stage_wall.get(stage, 0.0), wall / cycles_per_ms)

    n_stages = len(stage_wall)
    t_pipe = max(stage_wall.values())
    steps = (2 if mode == 'cwt' else 1) * n_stages
    return PipelineRecord(pool, mode, t_pipe, n_stages,
                          len({c for _, c in acc}), steps * t_pipe)


def throughput_ant_ms(rec: PipelineRecord) -> float:
    return 1.0 / rec.t_pipe if rec.t_pipe > 0 else 0.0         # 1 / T_pipe


def config_label(rec: PipelineRecord) -> str:
    return f'{rec.pool}\n{rec.mode.upper()}'


def print_metrics(records: List[PipelineRecord]) -> None:
    w = max([11] + [len(r.pool) for r in records]) + 2
    print(f'\n{"pool":<{w}}{"mode":<5}{"T_pipe[ms]":>12}'
          f'{"latency[ms]":>13}{"throughput[ant/ms]":>21}')
    for r in records:
        print(f'{r.pool:<{w}}{r.mode.upper():<5}{r.t_pipe:>12.4f}'
              f'{r.latency:>13.4f}{throughput_ant_ms(r):>21.4f}')


def plot_bars(records: List[PipelineRecord], title: str, out_path: Path) -> None:
    try:
        import matplotlib
        matplotlib.use('Agg')   # headless: write a PNG, never open a window
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.ticker import MaxNLocator
    except ImportError:
        print('Warning: matplotlib not installed; PNG skipped '
              '(pip install matplotlib).  Text table above still printed.')
        return

    order = {'ctt': 0, 'cwt': 1}
    recs = sorted(records, key=lambda r: (r.pool, order.get(r.mode, 99)))

    pools = sorted({r.pool for r in recs})
    cmap = LinearSegmentedColormap.from_list('bluered', ['#1f77b4', '#d62728'])
    colour = [cmap(pools.index(r.pool) / max(1, len(pools) - 1)) for r in recs]
    hatch = ['//' if r.mode == 'cwt' else '' for r in recs]
    labels = [config_label(r) for r in recs]
    xpos = list(range(len(recs)))

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.8), constrained_layout=True)

    def draw_decimal(ax, values, ylabel, title):
        """Plain bars with a 4-decimal label (used for latency, which is a time)."""
        bars = ax.bar(xpos, values, color=colour, edgecolor='white')
        for b, h in zip(bars, hatch):
            b.set_hatch(h)
        for x, v in zip(xpos, values):
            ax.text(x, v, f'{v:.4f}', ha='center', va='bottom', fontsize=9)
        ax.set_ylim(0, max(values) * 1.15 if values else 1)      # headroom for labels
        _finish(ax, ylabel, title)

    def draw_antennas(ax, values, ylabel, title):
        """Antennas are countable: draw floor(v) whole antennas solid + the fractional
        antenna as a lighter cap, headline the integer, keep the exact value."""
        for x, v, c, h in zip(xpos, values, colour, hatch):
            whole = math.floor(v)
            ax.bar(x, whole, color=c, edgecolor='white', hatch=h)
            frac = v - whole
            if frac > 1e-9:
                ax.bar(x, frac, bottom=whole, color=c, edgecolor='white',
                       hatch=h, alpha=0.30)
            # integer headline right on the bar, exact decimal just above it
            ax.annotate(f'{whole:d}', (x, v), textcoords='offset points', xytext=(0, 2),
                        ha='center', va='bottom', fontsize=13, fontweight='bold')
            ax.annotate(f'{v:.4f}', (x, v), textcoords='offset points', xytext=(0, 19),
                        ha='center', va='bottom', fontsize=8, color='0.35')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))     # whole antennas
        ax.set_ylim(0, max(values) * 1.24 if values else 1)       # room for both labels
        _finish(ax, ylabel, title)

    def _finish(ax, ylabel, title):
        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, axis='y', alpha=0.3)

    draw_decimal(axL, [r.latency for r in recs],
                 'latency [ms]', 'latency per execution  =  steps $\\times$ $T_{pipe}$')
    draw_antennas(axR, [throughput_ant_ms(r) for r in recs],
                  'throughput [antennas/ms]  (whole + partial)',
                  'throughput  =  1 / $T_{pipe}$')

    fig.suptitle(title, fontsize=13)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f'\nSaved plot -> {out_path}')


def figure_title(records: List[PipelineRecord], inputs: List[Path], tag: str) -> str:
    """One line naming what is on the axes: the run, and its size when they agree."""
    base = inputs[0].stem if len(inputs) == 1 else tag
    sizes = {r.n_clusters for r in records}
    return f'{base}  ({sizes.pop()}-cluster pipeline)' if len(sizes) == 1 else base


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('inputs', nargs='+', metavar='CSV',
                    help='One or more measured per-kernel CSVs '
                         '(parse_bench.py --csv), one config each.')
    ap.add_argument('--out', metavar='PNG',
                    help='Output PNG (default: <flavour>_bench_plots/'
                         '<TAG>_latency_throughput[_<file>].png next to the input, '
                         'the same directory parse_bench.py plots into).')
    ap.add_argument('--mode', choices=('ctt', 'cwt'),
                    help='Force the mode of every CSV given, for files whose name does '
                         'not carry a _ctt_ / _cwt_ tag.')
    ap.add_argument('--freq-hz', type=float, default=1e9, metavar='F',
                    help='Clock frequency in Hz, for cycles -> ms (default: 1e9).')
    ap.add_argument('--ctt', action='store_true',
                    help='Plot only the ctt (compute-then-transfer) mode.')
    ap.add_argument('--cwt', action='store_true',
                    help='Plot only the cwt (compute-while-transfer) mode.')
    args = ap.parse_args()

    wanted = {m for m, on in (('ctt', args.ctt), ('cwt', args.cwt)) if on} or {'ctt', 'cwt'}

    inputs = [Path(p) for p in args.inputs]
    missing = [p for p in inputs if not p.is_file()]
    if missing:
        raise SystemExit(f'No such file(s): {", ".join(str(p) for p in missing)}')

    records = [read_csv_record(p, args.mode, args.freq_hz) for p in inputs]
    records = [r for r in records if r.mode in wanted]
    if not records:
        raise SystemExit(f'No pipeline rows for mode(s) {sorted(wanted)} in '
                         f'{", ".join(str(p) for p in inputs)}')

    flavour = flavour_of(inputs[0].stem)
    tag = 'CEViT' if flavour == 'cevit' else 'CELERE'
    # Default alongside the input, in the directory parse_bench.py plots into, so a
    # run's figures land together.  One file names itself; several share one figure.
    out = Path(args.out) if args.out else (
        inputs[0].parent / f'{flavour}_bench_plots' /
        (f'{tag}_latency_throughput_{inputs[0].stem}.png' if len(inputs) == 1
         else f'{tag}_latency_throughput.png'))

    print_metrics(records)
    plot_bars(records, figure_title(records, inputs, tag), out)


if __name__ == '__main__':
    main()
