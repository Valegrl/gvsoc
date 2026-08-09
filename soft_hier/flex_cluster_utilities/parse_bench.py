#!/usr/bin/env python3
"""Aggregate the [BENCH] lines emitted by the mempool trace CSR (0x7d0) into a
per-kernel performance report.

Compute regions are scored against the op-issue peak of the unit that runs them
(TE, PE, or the LSU for the arithmetic-free patch permutations) and DMA regions
against their link's peak bandwidth (off-chip LPDDR, inter-cluster NoC,
cluster-local), so every row carries a utilization.

Every core writes CSR_TRACE around each region, so a run produces one line per
(core, region).  This groups them back into the layers of the pipeline.

    ./parse_bench.py log_bench.log --app main_164_geluTo4.c
    ./parse_bench.py log_bench.log --app main.c --csv out.csv
    ./parse_bench.py log_bench.log --raw        # no labels, just region indices
    ./parse_bench.py log_bench.log --app main.c --plot_ctt   # and the stage figure

The log carries no semantics -- a [BENCH] record identifies its region only by a
numeric index -- so the kernel names, transfer sizes and GEMM shapes are read out
of the app that produced it, the main.c emitted by cevit_pipeline.py.  That keeps
the report honest across regenerations: change the layer mix, the row count or a
hand-off size and the numbers follow, with nothing to keep in sync by hand.
Without --app the run is reported raw, by region index, since guessing a layout
would silently produce wrong utilizations rather than none.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from pathlib import Path

BENCH_RE = re.compile(
    r"^\[BENCH\] cycles=(?P<cycles>-?\d+) ns=(?P<ns>-?\d+) "
    r"start=(?P<start>\d+) end=(?P<end>\d+) region=(?P<region>\d+) "
    r"hart=(?P<hart>\d+) path=(?P<path>\S+)"
)
CLUSTER_RE = re.compile(r"^/chip/(cluster_\d+)/")

# --------------------------------------------------------------------------
# The app model
#
# The log carries no semantics: a [BENCH] record identifies its region only by a
# numeric index.  Everything else -- which kernel ran, what shape it was, how
# many bytes a DMA moved -- lives in the generated app, so that is what we read.
# cevit_pipeline.py emits one statement per line with a trailing comment naming
# the layer, which makes the file straightforward to walk.
#
# Two region sequences come out of each stage, because a core's region counter
# only advances for regions *that core* entered: the DM core runs the DMAs on
# top of the compute, the other cores run only the compute.
#
# The labels built here are the contract with work_model() below: '(N B)' sizes
# a DMA, 'MxNxK' sizes a GEMM, '(N el)' sizes an elementwise kernel, and
# dma_link() reads 'HBM' / '-> stage' out of a DMA label to pick the link.
# --------------------------------------------------------------------------

COMMENT_RE = re.compile(r"/\*.*?\*/", re.S)
DEFINE_RE = re.compile(r"^#define\s+(\w+)\s+(\d+)[UL]*", re.M)
STAGE_TITLE_RE = re.compile(r"^/\* stage (\d+) : (.*?) \*/", re.M)
# The compute-while-transfer generator passes the item parity in, so the
# signature is compute_stage_N(uint32_t buf) there and compute_stage_N(void) in
# the blocking one.
COMPUTE_FN_RE = re.compile(
    r"^static inline void compute_stage_(\d+)\([^)]*\)\s*\n\{(.*?)^\}", re.M | re.S)
# The DMA helpers the generator emits, newest spelling first.  Each brackets
# itself with start/stop_benchmark, so every CALL is a region -- including the
# four that a transpose issues from inside its loop.  dma_1d() is the older
# generator's contiguous helper, kept so its logs still parse.
DMA_CALLS = ("dma_2d_strided", "dma_2d", "dma_1d")
# "Skip Sum [add]: Y += X" -> ("Skip Sum", "add")
COMMENT_NAME_RE = re.compile(r"^(.*?)\s*\[(\w+)\]")

# One entry per baremetal kernel the generator can emit: how to pull its
# problem size out of the call, and how to spell the label work_model() wants.
#
# A buffer operand is a bare name in the blocking app (X_BUF) and a subscript in
# the compute-while-transfer one (X_BUF[buf]), so match both -- anything up to
# the comma that separates it from the next argument.
B = r"[^,]+"
KERNEL_SPECS = [
    (re.compile(rf"redmule_synch_parallel\(\s*{B},\s*{B},\s*{B},\s*"
                r"(\d+)u?,\s*(\d+)u?,\s*(\d+)u?"),
     lambda n, g: f"GEMM {n} {g[0]}x{g[1]}x{g[2]} (RedMulE)"),
    (re.compile(rf"layernorm_parallel_2x4_f16vec\(\s*{B},\s*{B},\s*(\d+)u?,\s*(\d+)u?"),
     lambda n, g: f"NORM {n} {g[0]}x{g[1]}"),
    (re.compile(rf"softmax_parallel_2x4_f16vec\(\s*{B},\s*{B},\s*(\d+)u?,\s*(\d+)u?"),
     lambda n, g: f"SMAX {n} {g[0]}x{g[1]}"),
    (re.compile(rf"axpy_f16vecp_local_unrolled4\({B},\s*{B},\s*{B},\s*(\d+)u?"),
     lambda n, g: f"AXPY {n} ({g[0]} el)"),
    (re.compile(rf"gelu_f16\(\s*{B},\s*(\d+)u?"),
     lambda n, g: f"GELU {n} ({g[0]} el)"),
    (re.compile(rf"relu_f16\(\s*{B},\s*(\d+)u?"),
     lambda n, g: f"RELU {n} ({g[0]} el)"),
    # patchify_f16(src, dst, Nf, Nt, C, Nh, Nw, cid, nt) and its inverse.  The
    # label carries the shape the kernel actually walks -- R output rows of
    # Nh*Nw elements -- since that is what reshape_work() costs; the grid dims
    # are only how the app spells it.
    (re.compile(rf"(un)?patchify_f16\(\s*{B},\s*{B},\s*"
                r"(\d+)u?,\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?"),
     lambda n, g: reshape_label(n, g)),
]


def reshape_label(name, g):
    """-> the label for a patchify_f16 / unpatchify_f16 call.

    g is ('un' or None, Nf, Nt, C, Nh, Nw).  A patch is Nh x Nw and the channel
    index C is part of the ROW index, so the token matrix is
    (Nf/Nh)*(Nt/Nw)*C rows of Nh*Nw -- the same element count either way, which
    is why the two directions cost the same and share one model.
    """
    nf, nt, c, nh, nw = (int(x) for x in g[1:])
    rows = (nf // nh) * (nt // nw) * c
    return f"RSHP {name} {rows}x{nh * nw} ({'unpatchify' if g[0] else 'patchify'})"


def _comment_name(text):
    """-> the human part of a trailing comment ('Skip Sum [add]: ...' -> 'Skip Sum')."""
    m = COMMENT_NAME_RE.match(text)
    return m.group(1) if m else text


def _uint(arg):
    """-> the value of a literal C argument ('656u' -> 656), or None if it is not one."""
    m = re.fullmatch(r"(\d+)[uUlL]*", arg.strip())
    return int(m.group(1)) if m else None


def call_args(text, fn):
    """-> the argument list of the first `fn(...)` in text, split at top level.

    The arguments carry casts and address arithmetic of their own
    ('(uint64_t)(uintptr_t)X_BUF + i * 41984u'), so the split has to track
    parenthesis depth rather than just cutting at every comma.
    """
    m = re.search(rf"\b{fn}\s*\(", text)     # \b keeps dma_2d off dma_2d_strided
    if not m:
        return None
    depth, args, cur = 1, [], ""
    for ch in text[m.end():]:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if not depth:
                break
        if depth == 1 and ch == ",":
            args.append(cur)
            cur = ""
        else:
            cur += ch
    return [a.strip() for a in args + [cur]]


def dma_bytes(fn, args):
    """-> the bytes one call of a DMA helper moves, or None if they are not literal.

    dma_1d/dma_2d carry the count as their third argument; the strided form
    moves `repeat` chunks of `size`, which is how a transpose is issued -- the
    permutation becomes a strided gather, not a flat memcpy.
    """
    idx = (2, 5) if fn == "dma_2d_strided" else (2,)
    if len(args) <= max(idx):
        return None
    n = 1
    for k in idx:
        v = _uint(args[k])
        if v is None:
            return None
        n *= v
    return n


class AppModel:
    """The region layout of one generated app, read from its main.c."""

    def __init__(self, path):
        self.path = path
        src = Path(path).read_text()
        bare = COMMENT_RE.sub(" ", src)     # comments carry digits of their own

        d = {m.group(1): int(m.group(2)) for m in DEFINE_RE.finditer(bare)}
        self.n_stages = d["N_STAGES"]
        self.n_iter = d.get("N_ITER", 1)
        self.din_bytes = d["CLUSTER_0_DIN_BYTES"]

        self.cid_stage = self._array(bare, "CID_STAGE")
        self.lanes = self._array(bare, "STAGE_LANES")
        self.dout = self._array(bare, "STAGE_DOUT_BYTES")

        # The compute-while-transfer app overlaps the hand-off with the compute:
        # the dm core *issues* the outbound DMA (dma_1d_issue / push_to_next_issue,
        # no benchmark bracket), computes alongside it, then drains it.  So the
        # hand-off leaves no region in the log at all and must not be modelled --
        # counting it would shift every region index of the dm core by one.  The
        # blocking app calls push_to_next(), which brackets each transfer.
        self.overlapped = "push_to_next_issue" in bare

        self.titles = {int(m.group(1)): m.group(2)
                       for m in STAGE_TITLE_RE.finditer(src)}
        self.bodies = {int(m.group(1)): self._body(m.group(2))
                       for m in COMPUTE_FN_RE.finditer(src)}

        self.stages = {}
        for sid in range(self.n_stages):
            dm, worker = self._stage_regions(sid)
            self.stages[sid] = {"name": self.titles.get(sid, "?"),
                                "dm": dm, "worker": worker}

    @staticmethod
    def _array(bare, name):
        m = re.search(rf"{name}\s*\[[^\]]*\]\s*=\s*\{{([^}}]*)\}}", bare)
        if not m:
            raise ValueError(f"no {name}[] in app source")
        return [int(x) for x in re.findall(r"(\d+)", m.group(1))]

    @staticmethod
    def _logical_lines(text):
        """-> the body, one C statement per entry.

        A call may be split over several source lines -- the strided DMA is --
        so lines are joined until their parentheses balance again.  Comments are
        kept (they name the region) but ignored while counting.
        """
        out, buf, depth = [], "", 0
        for line in text.splitlines():
            buf = line if not buf else f"{buf} {line.strip()}"
            code = COMMENT_RE.sub(" ", line)
            depth += code.count("(") - code.count(")")
            if depth <= 0:
                out.append(buf)
                buf, depth = "", 0
        if buf:
            out.append(buf)
        return out

    @staticmethod
    def _region_of(line, ctx, dm):
        """-> [(label, dm_only)] if this statement is a bracketed region, else []."""
        own = re.search(r"/\*\s*(.*?)\s*\*/", line)
        for fn in DMA_CALLS:
            args = call_args(line, fn)
            if args is None:
                continue
            # The strided DMA's comment sits on the line above the `if
            # (flex_is_dm_core())` that guards it, so fall back to that.
            text = own.group(1) if own else ctx["comment"]
            ctx["comment"] = None
            nbytes = dma_bytes(fn, args)
            size = "" if nbytes is None else f", {nbytes} B"
            # A layout copy stays inside the cluster: L1 -> L1.
            return [(f"DMA  {_comment_name(text) if text else 'copy'} "
                     f"(layout copy{size})", dm)]
        if not own:
            return []
        for rx, mk in KERNEL_SPECS:
            k = rx.search(line)
            if k:
                ctx["comment"] = None
                return [(mk(_comment_name(own.group(1)), k.groups()), dm)]
        return []

    @staticmethod
    def _scan(lines, i, dm, ctx):
        """-> ([(label, dm_only), ...], index past the block that starts at i).

        Walks the block structure rather than the lines, for two reasons: a
        region inside a `for` is entered once per trip and so is that many
        regions in the log (the transpose issues one DMA per head), and a
        `if (flex_is_dm_core())` block makes every region inside it dm-only,
        which a per-line test would miss when the guard is on a line of its own.
        """
        out = []
        while i < len(lines):
            line = lines[i]
            i += 1
            bare = COMMENT_RE.sub(" ", line).strip()
            if bare.startswith("}"):
                return out, i
            if not bare:                        # a comment and nothing else
                m = re.search(r"/\*\s*(.*?)\s*\*/", line)
                # Only a region comment ('name [tag]: ...') labels a region; the
                # layer headings the generator emits ('FFN[rows 1-3] x1') do not.
                if m and COMMENT_NAME_RE.match(m.group(1)):
                    ctx["comment"] = m.group(1)
                continue
            trips = 1
            if bare.startswith("for"):
                m = re.search(r"<\s*(\d+)", bare)
                trips = int(m.group(1)) if m else 1
            # the guard covers this statement or block only, never its siblings
            dm_here = dm or "flex_is_dm_core" in bare
            if bare.endswith("{"):
                body, i = AppModel._scan(lines, i, dm_here, ctx)
                out.extend(body * trips)
                continue
            out.extend(AppModel._region_of(line, ctx, dm_here))
        return out, i

    @staticmethod
    def _body(text):
        """-> [(label, dm_only), ...] for the regions of one compute_stage_N()."""
        regions, _ = AppModel._scan(AppModel._logical_lines(text), 0, False,
                                    {"comment": None})
        return regions

    def _transfer_regions(self, sid):
        """The DM core's hand-off at the end of a step -- see push_to_next()."""
        if self.overlapped:                     # issued async, never bracketed
            return []
        own = self.dout[sid]
        if sid == self.n_stages - 1:
            return [f"DMA  L1 -> HBM (stage output, {own} B)"]
        lanes, lanes_nx = self.lanes[sid], self.lanes[sid + 1]
        if lanes == lanes_nx:
            return [f"DMA  push -> stage {sid + 1} ({own} B)"]
        if lanes == 1:
            chunk = own // lanes_nx
            return [f"DMA  scatter -> stage {sid + 1} head {k} ({chunk} B)"
                    for k in range(lanes_nx)]
        return [f"DMA  gather -> stage {sid + 1} ({own} B)"]

    def handoff_label(self, sid):
        """The hand-off ONE cluster of a stage performs, as a DMA region label.

        Every cluster of the producer stage drives its own link, so each gets a
        row of its own -- a 4-lane gather is four clusters pushing 4 x dout in
        parallel, not one transfer.  dout is already per-cluster, so it is the
        right size for every shape: a scatter is one cluster fanning its whole
        dout out over its own link (lanes == 1 there), a gather is each lane
        sending its own dout, a push is one for one.
        """
        own, lanes = self.dout[sid], self.lanes[sid]
        if sid == self.n_stages - 1:
            return f"DMA  L1 -> HBM (stage output, {own} B)"
        kind = ("push" if lanes == self.lanes[sid + 1]
                else "scatter" if lanes == 1 else "gather")
        return f"DMA  {kind} -> stage {sid + 1} ({own} B)"

    @staticmethod
    def _disambiguate(labels):
        """Two identical regions in one stage are two rows, not one.

        Regions are keyed by label, so a stage that runs the same kernel on the
        same shape twice -- as the gelu split does -- would otherwise fold both
        executions into a single row spanning from the first to the last.  The
        suffix goes after the trailing ')', where it is out of the way of the
        size and shape patterns work_model() reads back out.
        """
        seen = defaultdict(int)
        out = []
        for label in labels:
            seen[label] += 1
            out.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
        return out

    def _stage_regions(self, sid):
        dm, worker = [], []
        # Stage 0 alone pulls its input from HBM, at the top of the compute phase.
        if sid == 0:
            dm.append(f"DMA  HBM -> L1 (stage input, {self.din_bytes} B)")
        for label, dm_only in self.bodies.get(sid, []):
            dm.append(label)
            if not dm_only:
                worker.append(label)
        dm.extend(self._transfer_regions(sid))
        return self._disambiguate(dm), self._disambiguate(worker)

    def stage_of(self, cid):
        return self.cid_stage[cid] if cid < len(self.cid_stage) else None

    def table_for(self, cid):
        """-> {'dm': [...], 'worker': [...]} for a cluster, or None if unmapped."""
        return self.stages.get(self.stage_of(cid))

    def title_of(self, cid):
        stage = self.stages.get(self.stage_of(cid))
        return stage["name"] if stage else "?"


class NoApp:
    """--raw, or no app given: every region falls back to its bare index."""

    n_iter = 1

    def stage_of(self, cid):
        return None

    def table_for(self, cid):
        return None

    def title_of(self, cid):
        return "?"


# --------------------------------------------------------------------------
# Utilization model -- OP/cycle
#
#     uti = achieved OP/cyc / peak OP/cyc = ops / (peak_ops * span)
#
# An "OP" is one MAC for the TE, one 2-lane f16 vector FP instruction for the
# PE and one halfword access for the LSU, so uti is each unit's occupancy
# against its own op-issue peak.  Note the OP definitions differ: uti is
# comparable across units as a percentage, but the raw OP/cyc columns are not
# additive.  Only the TE and PE have a FLOP reading; the LSU does no arithmetic.
#
# This is the same ratio the RedMulE model prints for itself
# (light_redmule.cpp:1328, ideal_runtime / measured_cycles), so a TE row can be
# checked directly against a `make runv` trace.
#
# uti is also the FLOP utilization, not just an occupancy proxy.  Both units
# carry the same factor of 2 from op to FLOP -- a TE MAC is 2 FLOPs, a PE
# vector op is 2 f16 lanes -- so it cancels:
#     TE:  2*MACs / (2*4096*span) = MACs/(4096*span) = uti
#     PE:  2*ops  / (2* 256*span) = ops /( 256*span) = uti
# Hence one column suffices; only the absolute FLOP/s rate is reported too.
# --------------------------------------------------------------------------

# TE: nb_redmule_tiles x (redmule_height x redmule_width) MACs per cycle,
# from configs/arch_tensorpool.py.  light_redmule.cpp:370 defines one engine's
# ideal runtime as M*N*K / (ce_height*ce_width); the 16 engines run concurrently
# on 1/16 of the GEMM each (confirmed against a RedMulE trace: uti = 0.264 for
# the 688x128x84 Out Proj).
RM_TILES, RM_H, RM_W = 16, 8, 32
TE_MAC_PER_CYC = RM_TILES * RM_H * RM_W         # 4096 MAC/cyc per cluster

# PE: num_core_per_cluster cores, one 2-lane f16 vector FP op per cycle each.
N_CORES = 256
PE_OPS_PER_CYC = N_CORES                        # 256 vecop/cyc per cluster

# LSU: the patch permutation kernels do no arithmetic at all -- their inner loop
# is scalar halfword loads and stores (p.lh / sh, and the mirror image for
# unpatchify), one memory instruction per element moved, and the grid-side
# strides are not unit so there is nothing to vectorise.  Rating them against an
# FP peak would be meaningless, so they get a unit of their own: one load/store
# issue slot per core per cycle.  An "OP" here is one halfword access, hence
# 2 bytes; there is no FLOP figure for this unit and none is printed.
LSU_OPS_PER_CYC = N_CORES                       # 256 halfword access/cyc
LSU_BYTES_PER_OP = 2                            # a halfword
# Moving one element is a load and a store -- the two sides of the permutation.
# One of the pair is the post-incrementing Xpulp form (p.lh on the strided side
# of patchify, p.sh on the strided side of unpatchify) and the other a plain
# immediate-offset lh / sh, so the direction does not change the count: both
# kernels issue exactly two memory instructions per element and no address
# update of their own.
LSU_OPS_PER_ELEM = 2

# The clock domain is 1 GHz (flex_cluster.py:339), so 1 cycle == 1 ns and
# FLOP-per-cycle reads directly as GFLOP/s.
CLOCK_GHZ = 1.0

# Peak FLOP rates per cluster.  A PE's 32-bit FPU does two f16 MACs per cycle
# (one vfmac.h), so 256 cores x 2 MAC x 2 FLOP = 1 TFLOP/s; the TE's 4096
# MAC/cyc x 2 FLOP = 8 TFLOP/s.  An 8:1 ratio.
PEAK_GFLOPS = {
    "TE": TE_MAC_PER_CYC * 2 * CLOCK_GHZ,               # 8192
    "PE": PE_OPS_PER_CYC * 2 * 2 * CLOCK_GHZ,           # 1024
}

# Achieved FLOPs are reported as a floor: every vector op is credited with its
# two f16 lanes, so an op that is a vfmac.h (4 FLOPs) counts half.  Kernels made
# of vfmac -- axpy is all of them, gelu 12 of 44 -- therefore read low.  The uti
# column is unaffected: it is issue-slot occupancy, not FLOPs.
FLOP_PER_OP = 2

# The bank-interleaved elementwise kernels (axpy, gelu) all share one addressing
# idiom, read off the disassembly of the app ELF:
#     i = 2*core_id*BANKING_FACTOR;  i < Len;  i += 2*NUM_BANKS
# with 8 f16 elements handled per iteration.  The binary shows `mhartid << 4`
# for the start and an 0x1000 stride, which pins BANKING_FACTOR = 8 and
# NUM_BANKS = 256*8 = 2048 for this build.
BANKING_FACTOR = 8
BANKED_START = 2 * BANKING_FACTOR               # 16 elements per core offset
BANKED_STRIDE = 2 * N_CORES * BANKING_FACTOR    # 4096 elements
BANKED_ELEMS = 8                                # elements touched per iteration

# --------------------------------------------------------------------------
# DMA bandwidth model -- B/cycle
#
# A DMA region moves a known number of bytes (the region label carries the
# count) over one of three paths, each with its own peak:
#
#   LPDDR off-chip main memory, DRAMSys LPDDR4 model              6.4 GB/s
#   NoC   cluster to cluster over the 512-bit link
#         (noc_link_width, arch_tensorpool.py:76)                 64  GB/s
#   L1    cluster-local layout copy, L1 -> L1                     unmeasured
#
# The off-chip node is called "hbm" throughout the arch config and the app
# labels (hbm_start_base, "DMA HBM -> L1"), but the memory behind it is the
# LPDDR4 DRAMSys model -- hence the label match on HBM and the LPDDR naming.
#
# At 1 GHz a cycle is a ns, so B/cyc reads directly as GB/s and the same ratio
# applies as for the compute units:
#     uti = bytes / (peak_B_per_cyc * span)
# The L1 copies have no peak to divide by, so they report an achieved B/cyc
# with no utilization.
# --------------------------------------------------------------------------

LPDDR_PEAK_BPC = 6.4                            # B/cyc == GB/s at 1 GHz
NOC_PEAK_BPC = 64.0
L1_PEAK_BPC = None                              # no figure for the local path

DMA_BYTES_RE = re.compile(r"(\d+) B\b")

GEMM_SHAPE_RE = re.compile(r"(\d+)x(\d+)x(\d+)")

# What work_model() returns: uti = ops / (peak_ops * span), and ideal is the
# cycle count that implies, ops / peak_ops.  For a DMA an "op" is one byte, so
# ops is the transfer size and peak_ops the link's B/cyc; ideal and ceiling are
# None when the link has no known peak.
Work = namedtuple("Work", "kind ideal ceiling ops peak_ops")


def dma_link(label):
    """-> (link name, peak B/cyc) for a DMA region label.

    Every row is one cluster's transfer over one link, so the peaks are the
    per-link ones: concurrent lanes are separate rows, never summed into one.
    """
    if "HBM" in label:                          # off-chip, see above
        return "LPDDR", LPDDR_PEAK_BPC
    if "-> stage" in label:                     # push / scatter / gather
        return "NoC", NOC_PEAK_BPC
    return "L1", L1_PEAK_BPC                    # reshape / transpose / split


def rowpair_ops(m_rows, ops_per_pair):
    """Total ops and imbalance ceiling for the `_parallel_2x4_f16vec` kernels.

    Both softmax and layernorm walk rows two at a time:
        for (i = core_id*2; i < M; i += numThreads*2)
    so the M/2 row-pairs are dealt out over the cores, and the busiest core runs
    ceil((M/2)/numThreads) of them.  The ceiling is the best utilization this
    distribution allows even with a perfect, stall-free core.
    """
    pairs = (m_rows + 1) // 2
    total = pairs * ops_per_pair
    per_core = -(-pairs // N_CORES)          # ceil
    ideal_balanced = total / PE_OPS_PER_CYC
    critical = per_core * ops_per_pair       # the busiest core's issue slots
    return total, (ideal_balanced / critical if critical else None)


def layernorm_ops(n):
    # Vector FP ops per row-pair.
    # pass 1 (sum, sum of squares): 12 per 4 columns
    # reduction: 4 vfadd + 6 vfmul + 2 vfsub + 2 vfsqrt = 14
    # pass 2 (normalize): 4 vfsub + 4 vfdiv per 4 columns
    return (n // 4) * 12 + 14 + (n // 4) * 8


def softmax_ops(n):
    # pass 1 (row max): 4 vfmax per 4 columns;  reduction: 2 vfmax
    # pass 2 (exp via 3rd-order Taylor + running sum): 4 vfsub + 28 + 4 vfadd
    # reduction: 2 vfadd;  pass 3 (divide by sum): 4 vfdiv per 4 columns
    return (n // 4) * 4 + 2 + (n // 4) * 36 + 2 + (n // 4) * 4


def pe_work(rows, cols, ops_fn):
    """Build a Work for a `_parallel_2x4_f16vec` kernel."""
    total, ceil = rowpair_ops(rows, ops_fn(cols))
    return Work("PE", total / PE_OPS_PER_CYC, ceil, total, PE_OPS_PER_CYC)


def banked_work(length, ops_per_iter):
    """Build a Work for a bank-interleaved elementwise kernel.

    Note this addressing covers only half of [0, length): a core starts at
    16*core_id and handles 8 elements before jumping a full 4096-element stride,
    so elements 16c+8 .. 16c+15 are never touched.  Both call sites pass the
    true element count (the ELF loads 0x2b000 = 176128 for the gelu of the
    688x256 FFN tensor), so both kernels leave half their tensor unprocessed --
    a bug in the kernels, not in this model.  What is counted here is what the
    cores actually execute.
    """
    iters = [max(0, -(-(length - BANKED_START * c) // BANKED_STRIDE))
             for c in range(N_CORES)]
    ops = sum(iters) * ops_per_iter
    critical = max(iters) * ops_per_iter
    ideal = ops / PE_OPS_PER_CYC
    return Work("PE", ideal, (ideal / critical if critical else None),
                ops, PE_OPS_PER_CYC)


def reshape_work(rows, row_elems):
    """Build a Work for patchify_f16 / unpatchify_f16.

    One halfword load and one halfword store per element, no arithmetic, so the
    unit is the LSU.  Both kernels deal the R output rows out over the cores
    (`for r = core_id; r < R; r += numThreads`) and each row is whole, so the
    busiest core owns ceil(R/N_CORES) of them -- that ratio is the imbalance
    ceiling.

    What is NOT counted: the four integer div/mod that decompose r into
    (fp, tp, c) once per row, and the loop overhead of the Nw tail (Nw % 4
    elements fall out of the unrolled block).  Both are real cycles the cores
    spend, so the reported uti is issue-slot occupancy of the memory path and
    reads low on a short Nw -- it is not a claim that the rest is idle.
    """
    ops_per_row = LSU_OPS_PER_ELEM * row_elems
    total = rows * ops_per_row
    critical = -(-rows // N_CORES) * ops_per_row        # busiest core's rows
    ideal = total / LSU_OPS_PER_CYC
    return Work("LSU", ideal, (ideal / critical if critical else None),
                total, LSU_OPS_PER_CYC)


def work_model(label):
    """-> Work(kind, ideal_cycles, imbalance_ceiling, ops, peak_ops) or None."""
    if label.startswith("DMA"):
        m = DMA_BYTES_RE.search(label)
        if not m:
            return None
        nbytes = int(m.group(1))
        link, peak = dma_link(label)
        return Work(f"DMA:{link}", None if peak is None else nbytes / peak,
                    None, nbytes, peak)

    if "RedMulE" in label:
        m = GEMM_SHAPE_RE.search(label)
        if not m:
            return None
        mm, nn, kk = (int(x) for x in m.groups())
        macs = mm * nn * kk
        return Work("TE", macs / TE_MAC_PER_CYC, 1.0, macs, TE_MAC_PER_CYC)

    if label.startswith("NORM"):
        m = GEMM_SHAPE_RE.search(label) or re.search(r"(\d+)x(\d+)", label)
        rows, cols = (int(x) for x in m.groups()[:2])
        return pe_work(rows, cols, layernorm_ops)

    if label.startswith("SMAX"):
        rows, cols = (int(x) for x in re.search(r"(\d+)x(\d+)", label).groups())
        return pe_work(rows, cols, softmax_ops)

    if label.startswith("AXPY"):
        # The app inlines axpy_f16vecp_local_unrolled4 (main+0x6b0 in the ELF:
        # `mhartid << 4` start, 8 lw + 4 vfmac.h + 4 sw, 0x1000-byte stride),
        # not the scalar axpy_f16p -- so it is bank-interleaved like gelu, with
        # AXPYF16VEC_UNROLLED4_LOOP doing 4 vfmac.h per 8-element iteration.
        return banked_work(int(re.search(r"(\d+) el", label).group(1)), 4)

    if label.startswith("GELU"):
        # gelu_f16 is fork-only, so the op count comes from the app ELF: its
        # loop body at 0x80002eac is 44 f16 vector FP ops (24 vfmul, 12 vfmac,
        # 4 vfadd, 4 vfdiv) per 8-element iteration.
        return banked_work(int(re.search(r"(\d+) el", label).group(1)), 44)

    if label.startswith("RSHP"):
        rows, cols = (int(x) for x in re.search(r"(\d+)x(\d+)", label).groups())
        return reshape_work(rows, cols)

    return None


class Region:
    """One labelled region within one cluster, across all cores that ran it."""

    def __init__(self, label, order):
        self.label = label
        self.order = order          # execution order within the cluster
        self.cycles = []
        self.start = None
        self.end = None

    def add(self, cycles, start, end):
        self.cycles.append(cycles)
        self.start = start if self.start is None else min(self.start, start)
        self.end = end if self.end is None else max(self.end, end)

    @property
    def span(self):
        """Wall-clock cost: first core in to last core out.

        The cores are barrier-synced at the region boundaries, so this -- not the
        per-core mean -- is what the layer actually costs the cluster.
        """
        return self.end - self.start

    @property
    def ncores(self):
        return len(self.cycles)


def parse(path):
    """-> {cluster: {hart: [(cycles, start, end, region_idx), ...]}}"""
    per_core = defaultdict(lambda: defaultdict(list))
    nbench = 0
    for line in open(path, errors="replace"):
        m = BENCH_RE.match(line)
        if not m:
            continue
        nbench += 1
        c = CLUSTER_RE.match(m.group("path"))
        if not c:
            print(f"warning: unrecognised path {m.group('path')}", file=sys.stderr)
            continue
        per_core[c.group(1)][int(m.group("hart"))].append(
            (int(m.group("cycles")), int(m.group("start")),
             int(m.group("end")), int(m.group("region")))
        )
    return per_core, nbench


def find_dm_core(cores):
    """The DM core is the one that ran strictly the most regions (it owns the DMAs)."""
    counts = {h: len(v) for h, v in cores.items()}
    top = max(counts.values())
    candidates = [h for h, n in counts.items() if n == top]
    return candidates[0] if len(candidates) == 1 else None


def build(per_core, app):
    """-> {cluster: [Region, ...]} ordered by first execution."""
    out = {}
    for cluster, cores in sorted(per_core.items()):
        cid = int(cluster.split("_")[1])
        table = app.table_for(cid)
        dm = find_dm_core(cores)

        # Validate the whole cluster before labelling any of it: a per-hart check
        # would leave a cluster half labelled and half raw, depending on the order
        # the harts happen to appear in the log.
        if table is not None:
            for hart, entries in cores.items():
                role = "dm" if hart == dm else "worker"
                if len(entries) != len(table[role]):
                    print(f"warning: {cluster} hart {hart} ran {len(entries)} "
                          f"regions, app has {len(table[role])} for a {role} core "
                          f"-- falling back to raw for this cluster "
                          f"(is this log from this app?)",
                          file=sys.stderr)
                    table = None
                    break

        regions = {}
        for hart, entries in cores.items():
            role = "dm" if hart == dm else "worker"
            labels = table[role] if table else None
            for cycles, start, end, idx in sorted(entries, key=lambda e: e[3]):
                if labels is not None:
                    key = labels[idx]
                else:
                    key = f"[{role}] region {idx}"
                r = regions.get(key)
                if r is None:
                    r = regions[key] = Region(key, start)
                r.add(cycles, start, end)
                r.order = min(r.order, start)
        out[cluster] = sorted(regions.values(), key=lambda r: r.order)
    return out


def fmt_pct(x, width):
    """x is a ratio; prints as a percentage, or '-' when it is unknown."""
    return f"{'-':>{width}}" if x is None else f"{100.0 * x:>{width - 1}.1f}%"


# A DMA cannot beat its link.  A row that does is not a fast transfer, it is a
# region whose span is not the transfer's duration: the engine reported done
# once the interconnect accepted the data, before it reached the far end.  This
# is worth catching rather than printing, because the number looks like a
# result -- a 116928 B store to LPDDR "at 54.7 B/cyc" was, on inspection of
# DRAMSys' own counters, a transfer that never reached the DRAM at all.
OVER_PEAK_MARK = "  <- over link peak"


def over_link_peak(w, span):
    """True if a DMA region claims a rate its link cannot deliver."""
    return (w is not None and w.kind.startswith("DMA")
            and w.peak_ops is not None and span > 0
            and w.ops / span > w.peak_ops)


# The rate columns: compute units are rated in OP/cyc, DMAs in B/cyc, and each
# row fills only its own pair.
RATE_HEAD = (f"{'OP/cyc':>8} {'peakOP/cyc':>10} {'B/cyc':>7} {'peakB/cyc':>9}")
RATE_EMPTY = f"{'-':>8} {'-':>10} {'-':>7} {'-':>9}"


def rate_cells(w, span):
    """The four rate columns for one region: OP/cyc, peak, B/cyc, peak."""
    rate = w.ops / span
    peak = "?" if w.peak_ops is None else f"{w.peak_ops:g}"
    if w.kind.startswith("DMA"):
        return f"{'-':>8} {'-':>10} {rate:>7.1f} {peak:>9}"
    # OP/cyc is an average over a region that is thousands of cycles long, so
    # its fractional part is far below the resolution of anything it is
    # compared against -- print it whole.  B/cyc keeps a decimal because the
    # LPDDR peak itself is 6.4.
    return f"{rate:>8.0f} {peak:>10} {'-':>7} {'-':>9}"


def add_handoff_regions(clusters, app, total_ns):
    """Give an overlapped app's unbracketed hand-offs the rows they never emit.

    The transfers are real work on the same links as everything else, so leaving
    them out of the tables would understate the run and hide the NoC entirely.
    They are issued async and never bracketed, so they write no [BENCH] record --
    but at N_ITER == 1 the schedule pins them exactly.  Stage s computes item 0
    at step 2s and pushes it at step 2s+1, and at that step compute_active is
    false for every stage in the machine: there is no item 1 for anyone to work
    on.  So step 2s+1 holds the push and nothing else, the global barrier at each
    end of it is the only other occupant, and the interval between the producer's
    last region and the consumer's first IS that step.  Nothing is estimated.

    Above N_ITER == 1 that stops holding -- the step would also carry a real
    overlapped compute -- so this runs only for the serialized case; report()
    says so and leaves the rows out otherwise.

    The last stage writes to HBM with no consumer to bound it, so its step comes
    from flex_timer: everything after the final region.
    """
    stage_win, stage_of = {}, defaultdict(list)
    for cluster, regions in clusters.items():
        if not regions:
            continue
        s = app.stage_of(int(cluster.split("_")[1]))
        if s is None:
            return
        lo = min(r.start for r in regions)
        hi = max(r.end for r in regions)
        stage_of[s].append(cluster)
        if s in stage_win:
            stage_win[s] = (min(stage_win[s][0], lo), max(stage_win[s][1], hi))
        else:
            stage_win[s] = (lo, hi)
    if not stage_win:
        return

    added = []
    for s in sorted(stage_win):
        if s + 1 in stage_win:
            start, end = stage_win[s][1], stage_win[s + 1][0]
        elif total_ns is not None:
            # The tail: flex_timer_start runs one barrier ahead of the first
            # region and flex_timer_end one barrier behind the last push, so the
            # count spans the same barrier-to-barrier step as the rows above.
            start = stage_win[s][1]
            end = min(lo for lo, _ in stage_win.values()) + total_ns
        else:
            continue
        if end <= start:
            continue
        # The step is delimited by a global barrier, so every lane of the stage
        # enters and leaves it together -- the window is the stage's, not each
        # cluster's own last region.  The lanes push concurrently over separate
        # links, which is why they are separate rows of dout bytes each and not
        # one row of their sum.
        for cluster in stage_of[s]:
            r = Region(app.handoff_label(s), start)
            r.add(end - start, start, end)
            clusters[cluster].append(r)
            added.append(cluster)

    for cluster in set(added):                  # the new rows are out of order
        clusters[cluster].sort(key=lambda r: r.order)


def overlap_note(app):
    """How an overlapped app's hand-off rows were arrived at, and what they mean."""
    print()
    if app.n_iter != 1:
        print(f"    NOTE: N_ITER is {app.n_iter}, so a push step also carries the compute")
        print("    it overlaps and its span is not the transfer's.  The hand-off rows are")
        print("    omitted -- the NoC totals above cover only the bracketed transfers.")
        return
    print("    NOTE: N_ITER is 1, so nothing overlaps.  Stage s computes item 0 at step")
    print("    2s and pushes it at step 2s+1, where compute_active is false for every")
    print("    stage -- there is no item 1 for anyone to work on.  The hand-off rows are")
    print("    those push steps, measured barrier to barrier; the app is running as")
    print("    compute-THEN-transfer, and the overlap needs N_ITER >= 2 to pay.")


def report(clusters, total_ns, app):
    kernel_totals = defaultdict(int)
    stage_span = {}
    impossible = []             # DMA rows whose span cannot be the transfer's

    for cluster, regions in clusters.items():
        cid = int(cluster.split("_")[1])
        stage = app.stage_of(cid)
        title = app.title_of(cid)
        busy = sum(r.span for r in regions)
        lo = min(r.start for r in regions)
        hi = max(r.end for r in regions)
        stage_span[cluster] = (lo, hi)

        print()
        print(f"=== {cluster}  (stage {stage}) -- {title}")
        print(f"    active {lo} -> {hi} cyc   sum of regions {busy} cyc")
        print(f"    {'region':<46} {'cores':>5} {'span':>9} "
              f"{'unit':>9} {RATE_HEAD} {'uti':>7}")
        for r in regions:
            n = r.ncores
            w = work_model(r.label)
            if w:
                cells = (f"{w.kind:>9} {rate_cells(w, r.span)} "
                         f"{fmt_pct(None if w.ideal is None else w.ideal / r.span, 7)}")
            else:
                cells = f"{'-':>9} {RATE_EMPTY} {'-':>7}"
            bad = over_link_peak(w, r.span)
            if bad:
                impossible.append((cluster, r.label, w.ops / r.span, w.peak_ops))
            print(f"    {r.label:<46} {n:>5} {r.span:>9} {cells}"
                  f"{OVER_PEAK_MARK if bad else ''}")
            # DMAs are split by link -- the three paths are worth telling apart.
            kernel_totals[w.kind if w and w.kind.startswith("DMA")
                          else r.label.split()[0]] += r.span

    print()
    print("=== utilization by kernel")
    print("    uti = achieved OP/cyc / peak OP/cyc.  OP is one MAC for the TE, one")
    print("          2-lane vector FP instruction for the PE, one halfword load or")
    print("          store for the LSU and one byte moved for a DMA, so the percentages")
    print("          are comparable but the raw OP/cyc columns are not additive.")
    print("          DMA B/cyc reads as GB/s at 1 GHz.")
    print(f"    {'kernel':<44}{'unit':>9} {'calls':>5} {'span':>8} "
          f"{RATE_HEAD} {'uti':>7}")
    agg = {}
    for cluster, regions in clusters.items():
        for r in regions:
            w = work_model(r.label)
            if not w:
                continue
            a = agg.setdefault(r.label, [w, 0, 0, 0.0, 0])
            a[1] += 1
            a[2] += r.span
            a[3] += w.ideal or 0.0
            a[4] += w.ops
    for label, (w, calls, span, ideal, ops) in sorted(
            agg.items(), key=lambda kv: -kv[1][2]):
        agg_w = w._replace(ops=ops)
        print(f"    {label:<44}{w.kind:>9} {calls:>5} {span:>8} "
              f"{rate_cells(agg_w, span)} "
              f"{fmt_pct(None if w.ideal is None else ideal / span, 7)}"
              f"{OVER_PEAK_MARK if over_link_peak(agg_w, span) else ''}")

    # One summary line per unit, in the order the units first appear above.
    peaks = {}
    for a in agg.values():
        peaks.setdefault(a[0].kind, a[0].peak_ops)
    for kind, peak_ops in peaks.items():
        rows = [a for a in agg.values() if a[0].kind == kind]
        span = sum(a[2] for a in rows)
        ideal = sum(a[3] for a in rows)
        ops = sum(a[4] for a in rows)
        if not span:
            continue
        if kind.startswith("DMA"):
            peak_s = "?" if peak_ops is None else f"{peak_ops:g}"
            uti_s = "-" if peak_ops is None else f"{100.0 * ideal / span:.1f}%"
            over = peak_ops is not None and ops / span > peak_ops
            print(f"    -> {kind} overall: {ops / span:.1f} of {peak_s} B/cyc "
                  f"= {uti_s}   ({ops} B in {span} cyc, "
                  f"{ops / span * CLOCK_GHZ:.1f} GB/s)"
                  f"{OVER_PEAK_MARK if over else ''}")
        elif kind in PEAK_GFLOPS:
            print(f"    -> {kind} overall: {ops / span:.0f} of {peak_ops} OP/cyc "
                  f"= {100.0 * ideal / span:.1f}%   "
                  # ">=" only on the PE: a TE MAC is exactly 2 FLOPs, while a
                  # PE vfmac.h is 4 and is credited with 2.
                  f"({'>=' if kind == 'PE' else ''}"
                  f"{FLOP_PER_OP * ops / span * CLOCK_GHZ:.0f} of "
                  f"{PEAK_GFLOPS[kind]:.0f} GFLOP/s)")
        else:
            # The LSU does no arithmetic, so there is no FLOP rate to quote; the
            # halfword traffic it sustains is the comparable figure.
            print(f"    -> {kind} overall: {ops / span:.0f} of {peak_ops} OP/cyc "
                  f"= {100.0 * ideal / span:.1f}%   "
                  f"({LSU_BYTES_PER_OP * ops / span * CLOCK_GHZ:.1f} GB/s of "
                  f"halfword traffic, no FLOPs)")

    if impossible:
        print()
        print("=== over link peak -- these rows do NOT measure their transfer")
        for cluster, label, rate, peak in impossible:
            print(f"    {cluster:<10} {label:<46} {rate:>7.1f} of {peak:g} B/cyc")
        print()
        print("    A DMA cannot beat its link, so the span of these regions is not how")
        print("    long the transfer took: the engine reported done once the")
        print("    interconnect accepted the data, before it reached the far end -- and")
        print("    possibly before it arrived at all.  Cross-check against the memory")
        print("    model's own counters: DRAMSys prints AVG BW and Total Time per")
        print("    controller at the end of the log, and AVG_BW x Total_Time is the")
        print("    bytes that really reached the DRAM.  Everything derived from these")
        print("    spans -- the row's rate, the unit total above, the stage's transfer")
        print("    time and the pipeline model below -- is a LOWER bound on the cost.")
        # stdout and stderr are separately buffered; without this the warning
        # lands in the middle of a table when the two are piped together.
        sys.stdout.flush()
        print(f"warning: {len(impossible)} DMA region(s) over their link peak; their "
              f"spans are not transfer durations (see the report)", file=sys.stderr)

    print()
    print("=== by kernel class (sum of spans over all clusters)")
    grand = sum(kernel_totals.values())
    for k, v in sorted(kernel_totals.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<10} {v:>10} cyc  {100.0 * v / grand:5.1f}%")
    print(f"    {'TOTAL':<10} {grand:>10} cyc")

    print()
    print("=== pipeline")
    for cluster, (lo, hi) in sorted(stage_span.items()):
        regions = clusters[cluster]
        busy = sum(r.span for r in regions)
        print(f"    {cluster}: window {hi - lo:>7} cyc, busy {busy:>7} cyc "
              f"({100.0 * busy / (hi - lo):5.1f}% of window)")

    # The lane clusters of a head-parallel stage run concurrently, so a stage
    # costs max(lane windows), not their sum.
    by_stage = defaultdict(list)
    for cluster, (lo, hi) in stage_span.items():
        cid = int(cluster.split("_")[1])
        stage = app.stage_of(cid)
        by_stage[-1 if stage is None else stage].append((lo, hi))
    print()
    print("    critical path (stages run back to back, lanes in parallel)")
    crit = 0
    for stage in sorted(by_stage):
        lanes = by_stage[stage]
        window = max(hi for _, hi in lanes) - min(lo for lo, _ in lanes)
        crit += window
        print(f"      stage {stage} x{len(lanes):<2} {window:>8} cyc")
    print(f"      {'sum':<11} {crit:>8} cyc")

    lo = min(lo for lo, _ in stage_span.values())
    hi = max(hi for _, hi in stage_span.values())
    print(f"      {'end to end':<11} {hi - lo:>8} cyc "
          f"({100.0 * crit / (hi - lo):.1f}% in regions, rest is barriers)")
    if total_ns is not None:
        print(f"      {'flex_timer':<11} {total_ns:>8} ns")

    if getattr(app, "overlapped", False):
        overlap_note(app)


def write_csv(clusters, path, app):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "stage", "order", "region", "unit", "cores",
                    "span_cyc", "min_cyc", "mean_cyc", "max_cyc",
                    "start_cyc", "end_cyc",
                    "ops", "ops_per_cyc", "peak_ops_per_cyc",
                    "bytes", "bytes_per_cyc", "peak_bytes_per_cyc",
                    "ideal_cyc", "uti", "ceiling"])
        for cluster, regions in clusters.items():
            cid = int(cluster.split("_")[1])
            stage = app.stage_of(cid)
            stage = "" if stage is None else stage
            for i, r in enumerate(regions):
                n = r.ncores
                m = work_model(r.label)
                if m:
                    # A row fills either the op columns or the byte columns:
                    # compute regions are rated in OP/cyc, DMAs in B/cyc
                    # (== GB/s at 1 GHz).
                    kind = m.kind
                    rate = [m.ops, f"{m.ops / r.span:.1f}",
                            "" if m.peak_ops is None else f"{m.peak_ops:g}"]
                    blank = ["", "", ""]
                    cells = (blank + rate if kind.startswith("DMA")
                             else rate + blank)
                    cells += ["" if m.ideal is None else f"{m.ideal:.1f}",
                              "" if m.ideal is None else f"{m.ideal / r.span:.4f}",
                              "" if m.ceiling is None else f"{m.ceiling:.4f}"]
                else:
                    kind = ""
                    cells = [""] * 9
                w.writerow([cluster, stage, i, r.label, kind, n, r.span,
                            min(r.cycles), sum(r.cycles) // n, max(r.cycles),
                            r.start, r.end, *cells])


# --------------------------------------------------------------------------
# Pipeline figure -- compute against transfer, per stage
#
# The same figure cevit_csv_plot.py draws from a written-out CSV, drawn here
# straight from the regions in memory: one x slot per pipeline stage, the
# clusters of a head-parallel stage side by side inside a shaded slot.  Each
# region is charged to compute or transfer by the unit work_model() gives it:
#
#   transfer -- DMA:NoC, DMA:LPDDR   (stage hand-offs and the HBM read/write)
#   compute  -- everything else: TE, PE and the DMA:L1 layout copies
#
# which mirrors how cevit_pipeline.py costs the model: reshape/split/transpose
# are local memcpys inside a cluster's compute, only the hand-off crosses a link.
#
# A lane's height is the SUM of its region spans -- exactly what the CSV route
# can express, so the two agree number for number.  Each bar is then subdivided
# into the regions that make it up, one segment per kernel and per DMA: one
# figure that answers both "how much of this stage is transfer" and "which
# kernels is the rest made of".  In CTT the segments run bottom to top in
# EXECUTION order; CWT splits them over its two bars, so there they are grouped
# by side and only ordered within each.
# --------------------------------------------------------------------------

TRANSFER_KINDS = ("DMA:NoC", "DMA:LPDDR")

# One colour per kernel class.  Compute is warm, the links are cool-green, so a
# subdivided bar still reads as compute against transfer at a glance.
LAYER_COLORS = {
    "GEMM":      "#d62728",
    "GELU":      "#ff9896",
    "NORM":      "#ff7f0e",
    "SMAX":      "#e377c2",
    "AXPY":      "#8c564b",
    "RELU":      "#c49c94",
    "RSHP":      "#9467bd",
    "DMA L1":    "#ffbb78",     # layout copy: local, hence compute-side warm
    "DMA NoC":   "#2ca02c",
    "DMA LPDDR": "#17becf",
}
# For an app with kernels this palette has not met: keyed by side, so a new
# compute kernel never gets a link colour and vice versa.
LAYER_FALLBACK = {
    False: ("#9467bd", "#bcbd22", "#c5b0d5", "#dbdb8d", "#7f7f7f"),
    True:  ("#98df8a", "#7fcdbb", "#9edae5", "#1f77b4", "#aec7e8"),
}

# How a class is spelled in the legend.
LAYER_LEGEND = {
    "DMA NoC": "DMA remote L1",
    "RSHP":    "RSHP",
}


@dataclass
class Layer:
    """One region of a lane, as a segment of its bar."""
    cls:      str       # legend class: the kernel for compute, the link for a DMA
    ms:       float
    transfer: bool
    din:      bool = False      # the unoverlappable `stage input` read


@dataclass
class Lane:
    """One physical cluster of a stage."""
    cid:         int        # physical cluster id (the N of cluster_N)
    name:        str        # cluster label as it appears in the report
    compute_ms:  float
    transfer_ms: float
    din_ms:      float = 0.0    # the `stage input` HBM -> L1 read, already INSIDE
                                # transfer_ms; see wall()
    layers:      list = field(default_factory=list)  # its regions, execution order

    def wall(self, overlap):
        """Wall of this cluster [ms].

        CWT overlaps a cluster's outbound hand-off with its compute, but NOT its
        stage input: the generated loop reads that with the blocking dma_1d and
        then holds the worker cores on flex_intra_cluster_sync() until it lands,
        so it is serialized in front of the compute.  It therefore sits outside
        the max -- matching cevit_pipeline.py::_overlap_wall.  Only stage 0 has
        one, so every other lane reduces to the plain max(compute, transfer).
        """
        if overlap:
            return self.din_ms + max(self.compute_ms, self.transfer_ms - self.din_ms)
        return self.compute_ms + self.transfer_ms


@dataclass
class Stage:
    """One pipeline stage: n_par lanes running in parallel."""
    idx:   int
    lanes: list = field(default_factory=list)

    @property
    def n_par(self):
        return len(self.lanes)

    def wall(self, overlap):
        """Stage wall = the slowest of its parallel lanes, not their sum."""
        return max(l.wall(overlap) for l in self.lanes)


def is_transfer(kind):
    """True = crosses a link, False = compute (an unmodelled region counts as compute)."""
    return kind is not None and kind.startswith(TRANSFER_KINDS)


def region_class(label, kind):
    """The legend class of one region: its link if it is a DMA, else its kernel."""
    if kind is not None and kind.startswith("DMA:"):
        return "DMA " + kind.split(":", 1)[1]       # DMA NoC / DMA LPDDR / DMA L1
    head = label.split()
    return head[0] if head else "?"


def build_stages(clusters, app, cycles_per_ms):
    """-> [Stage, ...] from the same {cluster: [Region]} the report walks."""
    stages = {}
    unmodelled = set()
    for cluster, regions in sorted(clusters.items()):
        cid = int(cluster.split("_")[1])
        sidx = app.stage_of(cid)
        if sidx is None:
            continue
        comp = xfer = din = 0.0
        layers = []
        for r in regions:
            w = work_model(r.label)
            if w is None:
                unmodelled.add(r.label.split()[0])
            kind = None if w is None else w.kind
            # The stage input is tracked apart because CWT cannot overlap it; its
            # sibling `L1 -> HBM (stage output)` shares the unit but IS overlapped,
            # so the label is what separates the two.
            is_din = (is_transfer(kind) and "stage input" in r.label.lower())
            if is_transfer(kind):
                xfer += r.span
                din += r.span if is_din else 0
            else:
                comp += r.span
            layers.append(Layer(region_class(r.label, kind), r.span / cycles_per_ms,
                                is_transfer(kind), is_din))
        stages.setdefault(sidx, Stage(idx=sidx)).lanes.append(Lane(
            cid         = cid,
            name        = cluster,
            compute_ms  = comp / cycles_per_ms,
            transfer_ms = xfer / cycles_per_ms,
            din_ms      = din / cycles_per_ms,
            layers      = layers,
        ))
    if unmodelled:
        print(f"warning: unmodelled region(s) {sorted(unmodelled)} counted as compute",
              file=sys.stderr)
    for st in stages.values():
        st.lanes.sort(key=lambda l: l.cid)
    return [stages[k] for k in sorted(stages)]


def plot_cluster_split(label, stages, overlap, out_path, ymax):
    """Two stacked panels of per-cluster wall time, on one shared y axis.

    One x position per PIPELINE STAGE.  A head-parallel stage owns several
    clusters, so its clusters are drawn SIDE BY SIDE inside the stage's slot
    (shaded, and labelled with their cluster ids) -- the picture then shows
    directly that the stage's wall is the height of one lane, not the sum.

    Serialized split (CTT) -> stacked bars   (wall = compute + transfer).
    Overlapped split (CWT) -> grouped bars   (wall = max(compute, transfer)).

    The top panel is the plain split: compute red against transfer green, the
    coarse answer to where a stage's time goes.  The bottom panel is the same
    bars, same heights, subdivided into the regions that make them up -- one
    solid, full-width segment per kernel and per DMA, coloured by class and
    stacked in execution order.  Read one for the balance, the other for what is
    behind it; the panels share both axes, so a bar sits directly under its own
    split.  Only the totals are shared, so the segments need not line up with the
    colour boundary above -- stage 0's inbound HBM read sits at the bottom of the
    lower bar, where the cluster ran it, and inside the green half of the upper.

    In the CWT top panel a lane's `stage input` read is drawn hatched as the
    pedestal both its compute bar and its overlapped hand-off bar stand on --
    that read is serialized ahead of the compute, so it enters the wall as
    din + max(compute, rest).  Only stage 0 has one; every other lane reduces to
    the plain grouped pair.  Below it is just another LPDDR segment.
    """
    import matplotlib
    matplotlib.use("Agg")           # headless: write PNGs, never open a window
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    n_stages = len(stages)
    tag      = "CWT" if overlap else "CTT"
    n_phys   = sum(st.n_par for st in stages)
    any_din  = any(ln.din_ms > 0.0 for st in stages for ln in st.lanes)

    # Colour every class up front, biggest first, so the legend reads in order of
    # cost and a class keeps its colour across the figures of a run.
    total, side = defaultdict(float), {}
    for st in stages:
        for ln in st.lanes:
            for l in ln.layers:
                total[l.cls] += l.ms
                side[l.cls] = l.transfer
    order, palette, fb = sorted(total, key=lambda c: -total[c]), {}, {True: 0, False: 0}
    for cls in order:
        colour = LAYER_COLORS.get(cls)
        if colour is None:                  # an app the palette has not met yet
            pool = LAYER_FALLBACK[side[cls]]
            colour = pool[fb[side[cls]] % len(pool)]
            fb[side[cls]] += 1
        palette[cls] = colour
    # the stage input is an off-chip read, so in the kernel panel it is an LPDDR
    # segment like any other
    din_colour = palette.get("DMA LPDDR", "green")

    # Both panels share the y axis: the bars are the same heights in both, so the
    # subdivision below lines up row for row with the split above.  The legends
    # sit outside the axes -- the lower one is a row per kernel class, which no
    # amount of headroom inside the plot would reliably clear.
    fig, (ax_top, ax_ker) = plt.subplots(
        2, 1, sharex=True, sharey=True,
        figsize=(max(5.0, 1.3 * n_phys + 2.5) + 2.6, 7.6))

    span = 0.82                       # width of a stage slot
    tick_pos, tick_lab = [], []

    def stack(ax, x, layers, bottom, width):
        """Draw one region-per-segment stack; -> the top it reaches.

        Every segment is solid and the full width of its bar: this panel answers
        only which kernel, and the panel above already carries the compute
        against transfer split.
        """
        for lay in layers:
            ax.bar(x, lay.ms, width=width, bottom=bottom, color=palette[lay.cls],
                   edgecolor="white", linewidth=0.4, zorder=2)
            bottom += lay.ms
        return bottom

    for s, st in enumerate(stages):
        p     = st.n_par
        slots = 2 * p if overlap else p            # bars in this stage's slot
        bw    = span / slots
        # keep a lone cluster's bar from ballooning to the full slot width
        draw  = min(bw * 0.9, 0.30 if overlap else 0.34)
        left  = s - span / 2

        if p > 1:
            for ax in (ax_top, ax_ker):
                ax.axvspan(left - 0.03, left + span + 0.03, color="0.92", zorder=0)
        for lane_i, lane in enumerate(st.lanes):
            comp_layers = [l for l in lane.layers if not l.transfer]
            xfer_layers = [l for l in lane.layers if l.transfer]
            if overlap:
                xmid = left + (2 * lane_i + 1.0) * bw
                # the stage input read is exposed: it delays BOTH the compute and
                # the overlapped hand-off, so both bars start on top of it
                rest = lane.transfer_ms - lane.din_ms
                if lane.din_ms > 0.0:
                    for x in (xmid - bw / 2, xmid + bw / 2):
                        ax_top.bar(x, lane.din_ms, width=draw, color="green",
                                   hatch="//", edgecolor="white", zorder=2)
                        # below it is just its kernel class, like any other segment
                        ax_ker.bar(x, lane.din_ms, width=draw, color=din_colour,
                                   edgecolor="white", linewidth=0.4, zorder=2)
                ax_top.bar(xmid - bw / 2, lane.compute_ms, width=draw,
                           bottom=lane.din_ms, color="red", zorder=2)
                ax_top.bar(xmid + bw / 2, rest, width=draw, bottom=lane.din_ms,
                           color="green", zorder=2)
                # the pedestal is already drawn, so the transfer stack is the rest
                # of the hand-offs standing on it
                stack(ax_ker, xmid - bw / 2, comp_layers, lane.din_ms, draw)
                stack(ax_ker, xmid + bw / 2, [l for l in xfer_layers if not l.din],
                      lane.din_ms, draw)
            else:
                xmid = left + (lane_i + 0.5) * bw
                ax_top.bar(xmid, lane.compute_ms, width=draw, color="red", zorder=2)
                ax_top.bar(xmid, lane.transfer_ms, width=draw,
                           bottom=lane.compute_ms, color="green", zorder=2)
                # CTT sums the two halves, so the bar's height does not depend on
                # the order of its segments: stack them as the cluster ran them,
                # which puts stage 0's inbound HBM read at the bottom where it
                # happened instead of on top with the outbound transfers.
                stack(ax_ker, xmid, lane.layers, 0.0, draw)
            # cluster id above its own bar(s), clear of the stage ticks below the axis
            for ax in (ax_top, ax_ker):
                ax.annotate(f"c{lane.cid}", xy=(xmid, lane.wall(overlap)),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, color="0.35")

        tick_pos.append(s)
        tick_lab.append(f"stage {st.idx}" + (f"\n({p} clusters || )" if p > 1 else ""))

    top_handles = [Patch(facecolor="red", label="compute"),
                   Patch(facecolor="green", label="transfer")]
    if overlap and any_din:
        top_handles.append(Patch(facecolor="green", hatch="//", edgecolor="white",
                                 label="stage input (not overlapped)"))

    # Grouped by side: the header rows say which half of the bar above each
    # colour belongs to, so the split survives having a colour per kernel.
    ker_handles = []
    for is_xfer, head in ((False, "compute"), (True, "transfer")):
        group = [c for c in order if side[c] is is_xfer]
        if not group:
            continue
        ker_handles.append(Patch(facecolor="none", edgecolor="none", label=f"{head}:"))
        ker_handles += [Patch(facecolor=palette[c], edgecolor="white",
                              label=LAYER_LEGEND.get(c, c)) for c in group]

    ax_top.set_ylim(0.0, ymax if ymax else
                    1.15 * max(st.wall(overlap) for st in stages))
    ax_ker.set_xticks(tick_pos)
    ax_ker.set_xticklabels(tick_lab)
    ax_ker.set_xlim(-0.6, n_stages - 0.4)
    for ax, panel, handles in ((ax_top, "compute vs transfer", top_handles),
                               (ax_ker, "by kernel", ker_handles)):
        ax.tick_params(axis="x", length=0, pad=6)
        ax.set_ylabel("wall time [ms]")
        ax.set_title(panel, fontsize="small", loc="left", color="0.35")
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0),
                  fontsize="small", frameon=False, handlelength=1.4,
                  borderaxespad=0.0)
    ax_ker.set_xlabel("pipeline stage  (clusters of a stage run in parallel)")
    fig.suptitle(f"{label} -- {tag}  |  {n_stages} stages over {n_phys} clusters")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def print_pipeline_model(label, stages, overlap):
    """Per-stage split, plus the bottleneck (throughput) and steady-state latency."""
    tag   = "CWT" if overlap else "CTT"
    model = ("max(compute, transfer)   [stage input: din + max(compute, rest)]"
             if overlap else "compute + transfer")
    # the exposed stage input only changes the arithmetic in CWT, so give it a
    # column there and leave the CTT table as it was
    din_col = overlap and any(l.din_ms > 0.0 for st in stages for l in st.lanes)
    print()
    print(f"=== pipeline model: {label} -- {tag}   (stage wall = {model})")
    print(f"    {'stage':>5}  {'par':>3}  {'clusters':<14} "
          f"{'compute':>9}  {'transfer':>9}"
          + (f"  {'of which in':>11}" if din_col else "")
          + f"  {'wall':>9}   [ms]")
    if din_col:
        print(f"    {'':>5}  {'':>3}  {'':<14} {'':>9}  {'':>9}  {'(exposed)':>11}")
    for st in stages:
        cids = ",".join(f"c{l.cid}" for l in st.lanes)
        # a stage's lanes are replicas; report the slowest one, which sets the wall
        slow = max(st.lanes, key=lambda l: l.wall(overlap))
        print(f"    {st.idx:>5}  {st.n_par:>3}  {cids:<14} "
              f"{slow.compute_ms:>9.4f}  {slow.transfer_ms:>9.4f}"
              + (f"  {slow.din_ms:>11.4f}" if din_col else "")
              + f"  {slow.wall(overlap):>9.4f}")

    bottleneck = max(stages, key=lambda st: st.wall(overlap))
    t_pipe     = bottleneck.wall(overlap)
    # Steady state: the per-step global barrier makes every stage cost the
    # bottleneck period, and an item is resident one step per stage in CTT, two in
    # CWT (whose double-buffered schedule offsets consecutive stages by two steps).
    steps      = (2 if overlap else 1) * len(stages)
    print()
    print(f"    bottleneck  = stage {bottleneck.idx}  ->  T_pipe = "
          f"{t_pipe:.6g} ms  ({1e3 / t_pipe:.6g} items/s)")
    print(f"    latency     = {steps * t_pipe:.6g} ms  "
          f"({steps} steps x T_pipe, at regime)")


def make_plots(clusters, app, args):
    """Render whichever of --plot_ctt / --plot_cwt was asked for."""
    wanted = [(False, args.plot_ctt), (True, args.plot_cwt)]
    wanted = [(ov, path) for ov, path in wanted if path is not None]
    if not wanted:
        return
    if not isinstance(app, AppModel):
        print("error: --plot_ctt/--plot_cwt need --app (and not --raw): without the "
              "app there is no stage layout to plot", file=sys.stderr)
        return
    cycles_per_ms = args.freq_hz / 1e3
    stages = build_stages(clusters, app, cycles_per_ms)
    if not stages:
        print("error: no cluster mapped to a stage; nothing to plot", file=sys.stderr)
        return

    log = Path(args.log)
    label = args.plot_label or log.stem
    for overlap, path in wanted:
        tag = "CWT" if overlap else "CTT"
        if overlap != app.overlapped:
            print(f"warning: --plot_{tag.lower()} but the app has a "
                  f"{'compute-while-transfer' if app.overlapped else 'blocking'} "
                  f"hand-off; drawing it as asked", file=sys.stderr)
        print_pipeline_model(label, stages, overlap)
        out = (Path(path) if path else
               log.parent / "cevit_bench_plots" / f"CEViT_{tag}_split_{log.stem}.png")
        try:
            plot_cluster_split(label, stages, overlap, out, args.ymax)
        except ImportError:
            print("\nwarning: matplotlib not installed; plot skipped "
                  "(pip install matplotlib).", file=sys.stderr)
            return
        print(f"\n    [plot] {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log", help="GVSoC run log containing [BENCH] lines")
    ap.add_argument("--app", metavar="FILE",
                    help="main.c of the app that produced the log; supplies the "
                         "region labels, transfer sizes and kernel shapes")
    ap.add_argument("--raw", action="store_true",
                    help="do not label regions, even if --app is given")
    ap.add_argument("--csv", metavar="FILE", help="also write a CSV of the table")
    ap.add_argument("--plot_ctt", nargs="?", const="", metavar="PNG",
                    help="draw the compute-then-transfer figure (stacked bars): one "
                         "slot per stage, the clusters of a head-parallel stage side "
                         "by side.  Needs --app.  Default path is "
                         "cevit_bench_plots/CEViT_CTT_split_<log>.png next to the log")
    ap.add_argument("--plot_cwt", nargs="?", const="", metavar="PNG",
                    help="same figure for the compute-while-transfer flavour "
                         "(grouped bars, with the unoverlappable stage input drawn "
                         "hatched under both)")
    ap.add_argument("--ymax", type=float, metavar="MS",
                    help="fixed y-axis limit in ms (default: auto-scale; use 0.3 to "
                         "match the cevit_pipeline.py --plot figures)")
    ap.add_argument("--plot-label", metavar="NAME",
                    help="title label for the figure (default: the log file stem)")
    ap.add_argument("--freq-hz", type=float, default=CLOCK_GHZ * 1e9, metavar="F",
                    help="clock frequency in Hz for cycles -> ms (default: 1e9)")
    args = ap.parse_args()

    per_core, nbench = parse(args.log)
    if not nbench:
        print(f"no [BENCH] lines found in {args.log}", file=sys.stderr)
        return 1
    print(f"{nbench} [BENCH] records over {len(per_core)} clusters")

    if args.raw or not args.app:
        app = NoApp()
        if not args.raw:
            print("no --app given: reporting raw region indices, no utilization",
                  file=sys.stderr)
    else:
        app = AppModel(args.app)
        print(f"app {args.app}: {app.n_stages} stages, "
              f"{len(app.cid_stage)} clusters, "
              f"{'compute-while-transfer' if app.overlapped else 'blocking'} hand-off")
        if app.overlapped:
            print("note: the inter-stage hand-offs are issued async and never "
                  "bracketed, so they write no [BENCH] record; at N_ITER 1 each "
                  "one owns a step of its own, so the step is reported as the "
                  "transfer's region", file=sys.stderr)
        # Every stage repeats per item, so a second item would reuse the region
        # labels and silently fold two executions into one row.
        if app.n_iter != 1:
            print(f"warning: app streams {app.n_iter} items; regions repeat per "
                  f"item and are merged per label", file=sys.stderr)

    total_ns = None
    for line in open(args.log, errors="replace"):
        if "[Performance Counter]" in line:
            m = re.search(r"(\d+) ns", line)
            if m:
                total_ns = int(m.group(1))

    clusters = build(per_core, app)
    # Only the serialized case pins the unbracketed hand-offs to a step of their
    # own; see add_handoff_regions().
    if getattr(app, "overlapped", False) and app.n_iter == 1:
        add_handoff_regions(clusters, app, total_ns)
    report(clusters, total_ns, app)
    if args.csv:
        write_csv(clusters, args.csv, app)
        print(f"\nwrote {args.csv}")
    make_plots(clusters, app, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
