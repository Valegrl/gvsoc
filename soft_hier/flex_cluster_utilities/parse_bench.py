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
    ./parse_bench.py celere.log --app celere.c --celere      # the celere generator

Two generators emit apps this reads, and they lay out their stages differently,
so each gets a model of its own -- AppModel for cevit_pipeline.py and CelereApp
for celere_pipeline.py.  --cevit and --celere pick one; without either, the
flavour is detected from the tables the app source declares, so existing command
lines keep working unchanged.  --celere also tolerates a PARTIAL log: a run that
hung, was killed, or is still going ends mid-stage, and its last clusters hold a
prefix of their region table rather than all of it.  Those are labelled as far
as they got and named in a note, instead of being dropped to raw -- their
missing regions are simply absent, so their totals are lower bounds.

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


def cluster_id(name):
    """-> the N of 'cluster_N'.

    Also the sort key everywhere clusters are listed: the names sort as strings
    otherwise, which puts cluster_10 ahead of cluster_2 and scatters a stage's
    blocks through the report -- a 14-cluster run reads as if the stages had
    been mis-assigned when only the ordering is wrong.
    """
    return int(name.split("_")[1])

# --------------------------------------------------------------------------
# The app model
#
# A [BENCH] record names its region only by index, so kernel, shape and DMA size
# are read out of the generated app instead.  Each stage yields two region
# sequences: the DM core's (compute + DMAs) and the worker cores' (compute only).
#
# Label format is the contract with work_model(): '(N B)' sizes a DMA, 'MxNxK' a
# GEMM, '(N el)' an elementwise kernel; dma_link() reads the link out of it.
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
# itself, so every CALL is a region.  dma_1d() is the older generator's
# contiguous helper, kept so its logs still parse.
DMA_CALLS = ("dma_2d_strided", "dma_2d", "dma_1d")
# "Skip Sum [add]: Y += X" -> ("Skip Sum", "add")
COMMENT_NAME_RE = re.compile(r"^(.*?)\s*\[(\w+)\]")
# Matches the whole off-chip helper family: hbm_addr() for the default edge,
# hbm_<edge>() for a specific one.  A call using any of them is an LPDDR leg,
# not a cluster-local layout copy.
HBM_CALL_RE = re.compile(r"\bhbm_\w*\s*\(")

# One entry per baremetal kernel the generator can emit: how to pull its problem
# size out of the call, and how to spell the label work_model() wants.  A buffer
# operand is X_BUF or X_BUF[buf], so B matches anything up to the next comma.
B = r"[^,]+"
KERNEL_SPECS = [
    (re.compile(rf"redmule_synch_parallel\(\s*{B},\s*{B},\s*{B},\s*"
                r"(\d+)u?,\s*(\d+)u?,\s*(\d+)u?"),
     lambda n, g, t: f"GEMM {n} {g[0]}x{g[1]}x{g[2]} (RedMulE)"),
    (re.compile(rf"layernorm_parallel_2x4_f16vec\(\s*{B},\s*{B},\s*(\d+)u?,\s*(\d+)u?"),
     lambda n, g, t: f"NORM {n} {g[0]}x{g[1]}"),
    (re.compile(rf"softmax_parallel_2x4_f16vec\(\s*{B},\s*{B},\s*(\d+)u?,\s*(\d+)u?"),
     lambda n, g, t: f"SMAX {n} {g[0]}x{g[1]}"),
    (re.compile(rf"axpy_f16vecp_local_unrolled4\({B},\s*{B},\s*{B},\s*(\d+)u?"),
     lambda n, g, t: f"AXPY {n} ({g[0]} el)"),
    (re.compile(rf"gelu_f16\(\s*{B},\s*(\d+)u?"),
     lambda n, g, t: f"GELU {n} ({g[0]} el)"),
    (re.compile(rf"relu_f16\(\s*{B},\s*(\d+)u?"),
     lambda n, g, t: f"RELU {n} ({g[0]} el)"),
    # patchify_f16(src, dst, Nf, Nt, C, Nh, Nw, cid, nt) and its inverse.  The
    # label carries the shape the kernel walks (R rows of Nh*Nw), not the grid
    # dims, since that is what reshape_work() costs.
    (re.compile(rf"(un)?patchify_f16\(\s*{B},\s*{B},\s*"
                r"(\d+)u?,\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?"),
     lambda n, g, t: reshape_label(n, g)),
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


def _comment_parts(text):
    """-> (name, tag) of a trailing comment: 'Skip Sum [add]: ...' -> ('Skip Sum', 'add').

    The tag is the generator's own name for the kernel class, so it -- not the
    C function that happens to implement the region -- is what a label should
    lead with.  It matters for the stand-ins: several unrelated kernels are
    emitted as the same mempool_wait(), and only the tag tells them apart.
    Returns tag None when the comment carries no bracket.
    """
    m = COMMENT_NAME_RE.match(text)
    return (m.group(1), m.group(2)) if m else (text, None)


def _comment_name(text):
    """-> the human part of a trailing comment ('Skip Sum [add]: ...' -> 'Skip Sum')."""
    return _comment_parts(text)[0]


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

    flavour = "cevit"           # names the default plot directory and file stem
    plot_tag = "CEViT"

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

        # In the compute-while-transfer app the dm core issues the outbound DMA
        # unbracketed and drains it later, so the hand-off leaves no region in the
        # log and must not be modelled.  The blocking app brackets each transfer.
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
    def _region_of(line, ctx, dm, specs=None):
        """-> [(label, dm_only)] if this statement is a bracketed region, else [].

        `specs` selects the kernel table; the default is the cevit one, and
        CelereApp passes its own superset.
        """
        specs = KERNEL_SPECS if specs is None else specs
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
            # A layout copy stays inside the cluster (L1 -> L1); a call naming an
            # hbm_*() window is an LPDDR leg, which dma_link() must see as one.
            kind = "layout copy"
            if HBM_CALL_RE.search(args[0]):
                kind = "L1 -> LPDDR"
            elif len(args) > 1 and HBM_CALL_RE.search(args[1]):
                kind = "LPDDR -> L1"
            return [(f"DMA  {_comment_name(text) if text else 'copy'} "
                     f"({kind}{size})", dm)]
        if not own:
            return []
        for rx, mk in specs:
            k = rx.search(line)
            if k:
                ctx["comment"] = None
                name, tag = _comment_parts(own.group(1))
                return [(mk(name, k.groups(), tag), dm)]
        return []

    @staticmethod
    def _scan(lines, i, dm, ctx, specs=None):
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
                body, i = AppModel._scan(lines, i, dm_here, ctx, specs)
                out.extend(body * trips)
                continue
            out.extend(AppModel._region_of(line, ctx, dm_here, specs))
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


# --------------------------------------------------------------------------
# The celere app model
#
# celere_pipeline.py emits a different app shape from cevit_pipeline.py:
#
#   * stages numbered 1..N_STAGES (slot 0 of every table is dead), not from 0;
#   * compute_stage_N() is noinline, not `static inline void`;
#   * the hand-off is a table walk over STAGE_XFER, whose mode and the two
#     stages' lane counts set how many bracketed DMAs a stage issues;
#   * off-chip legs come from LPDDR_IN_XFER / LPDDR_OUT_XFER, one bracketed DMA
#     per entry; either range may be empty for a stage;
#   * a tiled last stage (STAGE_TILES > 1) drains itself and main() issues no
#     push for it.
#
# All of these DMAs bracket themselves, so the DM core's region indices run
# ahead of the worker cores' -- hence the dm/worker split in the region tables.
# --------------------------------------------------------------------------

CELERE_COMPUTE_FN_RE = re.compile(
    r"^static\s+void\s+__attribute__\(\(noinline\)\)\s+compute_stage_(\d+)\s*"
    r"\([^)]*\)\s*\n\{(.*?)^\}", re.M | re.S)
# `{ 1122304u, 4096u, XF_STRIDED }` -- two counts and a symbolic mode.
CELERE_XFER_RE = re.compile(r"\{\s*(\d+)u?\s*,\s*(\d+)u?\s*,\s*(XF_\w+)\s*\}")
# The lpddr_xfer_t rows are all numeric: lpddr_off, l1_off, lane_step, chunk,
# lpddr_stride, l1_stride, reps.
CELERE_LPDDR_RE = re.compile(r"\{([^}]*)\}")
# permute_PN_f16's own comment sizes it: "105216 blocks x 8 elements".
CELERE_PERM_SIZE_RE = re.compile(r"(\d+)\s+blocks\s+x\s+(\d+)\s+elements")
CELERE_PERM_FN_RE = re.compile(
    r"^static void (permute_\w+)\([^)]*\)[^{]*\{(.*?)^\}", re.M | re.S)

XF_CONTIG, XF_STRIDED, XF_RESPLIT = "XF_CONTIG", "XF_STRIDED", "XF_RESPLIT"

# celere's extra kernels, on top of the ones both generators share.
#
# The FFT and rope stand-ins are unscored: work_model() returns None and the row
# prints '-' rather than invent a peak.  The permutations ARE scored -- they are
# arithmetic-free load/store loops, so they reuse reshape_work() via RSHP.
CELERE_KERNEL_SPECS = KERNEL_SPECS + [
    # mempool_radix4_cfft_f16p(...).  Deliberately unsized: the kernel's header
    # is not in this tree, so its argument order is not established here -- and
    # the region is unscored either way.
    (re.compile(r"mempool_radix4_cfft_f16p\("), lambda n, g, t: f"FFT  {n}"),
    # mempool_wait(6012UL) -- a stand-in for a kernel that does not exist.  It
    # covers several (the FFTs and rope), so the label leads with the comment's
    # own [tag], not the call, to keep them as separate legend classes.
    (re.compile(r"mempool_wait\(\s*(\d+)[uUlL]*\s*\)"),
     lambda n, g, t: f"{_stand_in_label(n, t)} ({g[0]} cyc stand-in)"),
]


def _stand_in_label(name, tag):
    """-> 'FFT on dim i_2' / 'ROPE Rotate feature pairs ...' for a mempool_wait region.

    region_class() reads the class off the label's FIRST word, so the tag has to
    lead -- except where the name already opens with it ('FFT on dim i_2 [fft]'),
    which would otherwise print 'FFT FFT on dim i_2'.  'IFFT' is deliberately not
    treated as already-led: an inverse transform belongs in the FFT class, and
    prefixing is what puts it there.
    """
    cls = (tag or "wait").upper()
    head = name.split(None, 1)[0] if name.split() else ""
    return name if head.upper() == cls else f"{cls} {name}"


def _celere_short(name):
    """'P1[fftshift + box grouping]' -> 'P1'.

    _comment_name() peels a trailing '[tag]' of a single word, which is how the
    compute kernels spell themselves ('FFT on dim i_2 [fft]').  The
    permutations put a whole phrase in the bracket instead, so it survives that
    and has to come off here to keep the label to a readable width.
    """
    return name.split("[", 1)[0].strip() or name


class CelereApp:
    """The region layout of one generated celere app, read from its .c."""

    overlapped = False          # every hand-off is bracketed; see the note above
    flavour = "celere"          # names the default plot directory and file stem
    plot_tag = "CELERE"

    def __init__(self, path):
        self.path = path
        src = Path(path).read_text()
        bare = COMMENT_RE.sub(" ", src)

        d = {m.group(1): int(m.group(2)) for m in DEFINE_RE.finditer(bare)}
        self.defines = d
        self.n_stages = d["N_STAGES"]
        self.n_iter = d.get("N_ITER", 1)

        self.cid_stage = AppModel._array(bare, "CID_STAGE")
        self.cid_lane = AppModel._array(bare, "CID_LANE")
        self.lanes = AppModel._array(bare, "STAGE_LANES")
        self.tiles = AppModel._array(bare, "STAGE_TILES")
        self.xfer_first = AppModel._array(bare, "STAGE_XFER_FIRST")
        self.in_first = AppModel._array(bare, "LPDDR_IN_FIRST")
        self.out_first = AppModel._array(bare, "LPDDR_OUT_FIRST")
        self.xfer = self._xfer_table(bare)
        self.lpddr_in = self._lpddr_table(bare, "LPDDR_IN_XFER")
        self.lpddr_out = self._lpddr_table(bare, "LPDDR_OUT_XFER")

        # The kernel table gains one entry per permutation the app carries, each
        # closing over the size read out of that function's own comment.
        specs = list(CELERE_KERNEL_SPECS)
        for name, rows, cols in self._permutes(src):
            specs.append((re.compile(rf"\b{name}\("),
                          lambda n, g, t, r=rows, c=cols:
                              f"RSHP {_celere_short(n)} {r}x{c}"))
        self.specs = specs

        self.titles = {int(m.group(1)): m.group(2)
                       for m in STAGE_TITLE_RE.finditer(src)}
        # Sizes reach the DMA helpers as #defines (TILE_OUT_BYTES), and
        # dma_bytes() only reads literals, so fold them in before scanning.
        expanded = self._expand_defines(src, d)
        self.bodies = {int(m.group(1)): self._body(m.group(2), specs)
                       for m in CELERE_COMPUTE_FN_RE.finditer(expanded)}

        self.stages = {}
        for sid in range(1, self.n_stages + 1):
            dm, worker = self._stage_regions(sid)
            self.stages[sid] = {"name": self.titles.get(sid, "?"),
                                "dm": dm, "worker": worker}

    # -- source tables ----------------------------------------------------

    @staticmethod
    def _expand_defines(src, d):
        """Substitute the integer #defines so the size regexes see literals."""
        if not d:
            return src
        rx = re.compile(r"\b(" + "|".join(map(re.escape, sorted(d, key=len,
                                                                reverse=True)))
                        + r")\b")
        return rx.sub(lambda m: str(d[m.group(1)]), src)

    @staticmethod
    def _xfer_table(bare):
        m = re.search(r"STAGE_XFER\s*\[[^\]]*\]\s*=\s*\{(.*?)\n\}", bare, re.S)
        if not m:
            raise ValueError("no STAGE_XFER[] in app source")
        return [(int(a), int(b), mode)
                for a, b, mode in CELERE_XFER_RE.findall(m.group(1))]

    @staticmethod
    def _lpddr_table(bare, name):
        """-> [bytes_moved, ...], one entry per lpddr_xfer_t row (chunk * reps)."""
        m = re.search(rf"{name}\s*\[[^\]]*\]\s*=\s*\{{(.*?)\n\}}", bare, re.S)
        if not m:
            return []
        out = []
        for row in CELERE_LPDDR_RE.findall(m.group(1)):
            vals = [int(x) for x in re.findall(r"(\d+)u?", row)]
            if len(vals) >= 7:
                out.append(vals[3] * vals[6])       # chunk * reps
        return out

    @staticmethod
    def _permutes(src):
        """-> [(fn_name, blocks, elems_per_block), ...] for the app's permutations."""
        out = []
        for m in CELERE_PERM_FN_RE.finditer(src):
            size = CELERE_PERM_SIZE_RE.search(m.group(2))
            if size:
                out.append((m.group(1), int(size.group(1)), int(size.group(2))))
        return out

    # -- region tables ----------------------------------------------------

    @staticmethod
    def _fuse_bare_guards(lines):
        """Fold `if (flex_is_dm_core())` into the single statement it guards.

        AppModel._scan() only carries the guard into a following `{` block, so
        the braceless form celere uses for its tile drain would otherwise lose
        the dm-only marking and give the worker cores a region they never ran.
        """
        out, i = [], 0
        while i < len(lines):
            bare = COMMENT_RE.sub(" ", lines[i]).strip()
            if (re.fullmatch(r"if\s*\(\s*flex_is_dm_core\s*\(\s*\)\s*\)", bare)
                    and i + 1 < len(lines)):
                out.append(f"{lines[i]} {lines[i + 1]}")
                i += 2
            else:
                out.append(lines[i])
                i += 1
        return out

    @staticmethod
    def _body(text, specs):
        lines = CelereApp._fuse_bare_guards(AppModel._logical_lines(text))
        regions, _ = AppModel._scan(lines, 0, False, {"comment": None}, specs)
        return regions

    def _lpddr_regions(self, sid, table, first, arrow, what):
        if sid + 1 >= len(first):
            return []
        return [f"DMA  {arrow} ({what}, {table[e]} B)"
                for e in range(first[sid], min(first[sid + 1], len(table)))]

    def _handoff_regions(self, sid):
        """The bracketed DMAs push_to_next() issues for stage sid.

        Mirrors push_to_next() exactly: one region per dma_2d/dma_2d_strided
        call it makes, in the order it makes them.  main() skips the push for
        the last stage and for any tiled stage, which drains itself.
        """
        if sid >= self.n_stages or self.tiles[sid] != 1:
            return []
        lanes, lanes_nx = self.lanes[sid], self.lanes[sid + 1]
        out = []
        for e in range(self.xfer_first[sid], self.xfer_first[sid + 1]):
            nbytes, row_bytes, mode = self.xfer[e]
            own_src = nbytes // lanes
            own_dst = nbytes // lanes_nx
            if mode == XF_RESPLIT:
                # this lane sends slice k to lane k: lanes_nx transfers of an
                # equal share of what this cluster holds
                out += [f"DMA  resplit -> stage {sid + 1} lane {k} "
                        f"({own_src // lanes_nx} B)" for k in range(lanes_nx)]
            elif lanes == 1 and lanes_nx > 1:
                out += [f"DMA  scatter -> stage {sid + 1} lane {k} ({own_dst} B)"
                        for k in range(lanes_nx)]
            elif lanes > 1 and lanes_nx == 1:
                out.append(f"DMA  gather -> stage {sid + 1} ({own_src} B)")
            else:
                out.append(f"DMA  push -> stage {sid + 1} ({own_src} B)")
        return out

    def _stage_regions(self, sid):
        dm, worker = [], []
        dm += self._lpddr_regions(sid, self.lpddr_in, self.in_first,
                                  "LPDDR -> L1", "stage input")
        for label, dm_only in self.bodies.get(sid, []):
            dm.append(label)
            if not dm_only:
                worker.append(label)
        dm += self._lpddr_regions(sid, self.lpddr_out, self.out_first,
                                  "L1 -> LPDDR", "stage output")
        dm += self._handoff_regions(sid)
        return AppModel._disambiguate(dm), AppModel._disambiguate(worker)

    # -- the interface build()/report()/the plots use ----------------------

    def stage_of(self, cid):
        return self.cid_stage[cid] if cid < len(self.cid_stage) else None

    def table_for(self, cid):
        return self.stages.get(self.stage_of(cid))

    def title_of(self, cid):
        stage = self.stages.get(self.stage_of(cid))
        return stage["name"] if stage else "?"

    def handoff_label(self, sid):
        """Only add_handoff_regions() wants this, and it never runs for celere
        (every hand-off is bracketed, so overlapped is False)."""
        return f"DMA  push -> stage {sid + 1}"


def detect_flavour(path):
    """-> 'celere' / 'cevit' / None, from the tables the app source declares.

    The two generators lay out different tables, and either marker alone settles
    it.  A file carrying neither (or both) returns None so the caller can ask
    for the flag rather than guess and fail somewhere less obvious.
    """
    src = COMMENT_RE.sub(" ", Path(path).read_text())
    celere = "STAGE_XFER_FIRST" in src or "lpddr_pull" in src
    cevit = "STAGE_DOUT_BYTES" in src or "CLUSTER_0_DIN_BYTES" in src
    if celere == cevit:
        return None
    return "celere" if celere else "cevit"


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
# An "OP" is one MAC for the TE, one 2-lane f16 vector FP instruction for the PE
# and one halfword access for the LSU.  uti is comparable across units as a
# percentage, but the raw OP/cyc columns are not additive.  Only the TE and PE
# have a FLOP reading.
#
# uti doubles as FLOP utilization: the op->FLOP factor of 2 is the same for both
# units and cancels.  A TE row matches the ratio light_redmule.cpp prints, so it
# can be checked against a `make runv` trace.
# --------------------------------------------------------------------------

# TE: nb_redmule_tiles x (redmule_height x redmule_width) MACs per cycle, from
# configs/arch_tensorpool.py.  The 16 engines run concurrently on 1/16 of the
# GEMM each.
RM_TILES, RM_H, RM_W = 16, 8, 32
TE_MAC_PER_CYC = RM_TILES * RM_H * RM_W         # 4096 MAC/cyc per cluster

# PE: num_core_per_cluster cores, one 2-lane f16 vector FP op per cycle each.
N_CORES = 256
PE_OPS_PER_CYC = N_CORES                        # 256 vecop/cyc per cluster

# LSU: the patch permutation kernels do no arithmetic -- scalar halfword loads
# and stores, nothing vectorisable -- so they get a unit of their own: one
# load/store issue slot per core per cycle.  An "OP" is one halfword access
# (2 bytes); this unit has no FLOP figure.
LSU_OPS_PER_CYC = N_CORES                       # 256 halfword access/cyc
LSU_BYTES_PER_OP = 2                            # a halfword
# Moving one element is a load and a store, so both kernels issue exactly two
# memory instructions per element regardless of direction.
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

# Achieved FLOPs are a floor: every vector op counts 2 lanes, so a vfmac.h
# (4 FLOPs) counts half and vfmac-heavy kernels read low.  uti is unaffected --
# it is issue-slot occupancy, not FLOPs.
FLOP_PER_OP = 2

# The bank-interleaved elementwise kernels (axpy, gelu) share one addressing
# idiom, read off the app ELF:
#     i = 2*core_id*BANKING_FACTOR;  i < Len;  i += 2*NUM_BANKS
# with 8 f16 elements per iteration, which pins BANKING_FACTOR = 8.
BANKING_FACTOR = 8
BANKED_START = 2 * BANKING_FACTOR               # 16 elements per core offset
BANKED_STRIDE = 2 * N_CORES * BANKING_FACTOR    # 4096 elements

# --------------------------------------------------------------------------
# DMA bandwidth model -- B/cycle
#
# A DMA region moves a known number of bytes (the region label carries the
# count) over one of three paths, each with its own peak:
#
#   LPDDR off-chip main memory, per DRAMSys channel               from the log
#   NoC   cluster to cluster over the 512-bit link
#         (noc_link_width, arch_tensorpool.py:76)                 64  GB/s
#   L1    cluster-local layout copy, L1 -> L1                     unmeasured
#
# The off-chip node is spelled "hbm" in the arch config and app labels whatever
# DRAMSys model is behind it, so both spellings are matched.  Its peak is read
# per run from the DRAMSys footer by dram_peak_from_log(); the constant below is
# only the fallback for a log predating it, and --dram-peak overrides both.
#
# At 1 GHz a cycle is a ns, so B/cyc reads directly as GB/s:
#     uti = bytes / (peak_B_per_cyc * span)
# L1 copies have no peak, so they report an achieved B/cyc with no utilization.
# --------------------------------------------------------------------------

DRAM_PEAK_FALLBACK_BPC = 64.0                   # B/cyc == GB/s at 1 GHz
DRAM_PEAK_BPC = DRAM_PEAK_FALLBACK_BPC           # main() replaces this per run
NOC_PEAK_BPC = 64.0
L1_PEAK_BPC = None                              # no figure for the local path

# 'DRAMSysRecordable0.controller0  MAX BW:  512.00 Gb/s |  64.00 GB/s | 100.00 %'
DRAM_MAX_BW_RE = re.compile(r"MAX BW:.*?\|\s*([\d.]+) GB/s")

DMA_BYTES_RE = re.compile(r"(\d+) B\b")

GEMM_SHAPE_RE = re.compile(r"(\d+)x(\d+)x(\d+)")

# What work_model() returns: uti = ops / (peak_ops * span), ideal = ops/peak_ops.
# For a DMA an "op" is one byte.  ideal and ceiling are None with no known peak.
Work = namedtuple("Work", "kind ideal ceiling ops peak_ops")


def set_dram_peak(override, seen):
    """Fix the off-chip peak for this run, from --dram-peak or the log's footer.

    `seen` is every distinct 'MAX BW ... GB/s' the DRAMSys footer printed, one
    line per channel.  They are normally all the same figure; a run that mixes
    memories would make a single peak meaningless, so that is reported rather
    than averaged, and the largest is kept so the utilizations read as a floor.
    """
    global DRAM_PEAK_BPC
    if override is not None:
        DRAM_PEAK_BPC = override
        return
    if not seen:
        DRAM_PEAK_BPC = DRAM_PEAK_FALLBACK_BPC
        print(f"note: no DRAMSys footer in the log; assuming "
              f"{DRAM_PEAK_BPC:g} B/cyc per off-chip channel "
              f"(pass --dram-peak to set it)", file=sys.stderr)
        return
    if len(seen) > 1:
        print(f"warning: DRAMSys channels report different peaks "
              f"({', '.join(f'{p:g}' for p in sorted(seen))} GB/s); "
              f"using the largest, so the off-chip utilizations are a floor",
              file=sys.stderr)
    DRAM_PEAK_BPC = max(seen)


def dma_link(label):
    """-> (link name, peak B/cyc) for a DMA region label.

    Every row is one cluster's transfer over one link, so the peaks are the
    per-link ones: concurrent lanes are separate rows, never summed into one.
    """
    # 'HBM' is the cevit generator's spelling, 'LPDDR' the celere one; both name
    # the same off-chip link.
    if "HBM" in label or "LPDDR" in label:      # off-chip, see above
        return "LPDDR", DRAM_PEAK_BPC
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
        # The app inlines axpy_f16vecp_local_unrolled4, not the scalar axpy_f16p,
        # so it is bank-interleaved like gelu: 4 vfmac.h per 8-element iteration.
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


def build(per_core, app, partial_ok=False):
    """-> ({cluster: [Region, ...]} ordered by first execution, {cluster: (ran, of)}).

    The second value names the clusters that stopped early, and is empty unless
    partial_ok.  A run that hangs -- or one read while it is still going -- ends
    mid-stage, so the last clusters have a PREFIX of their region table rather
    than all of it.  With partial_ok those are labelled as far as they got
    instead of being dropped to raw, which is what makes a killed run readable;
    without it the exact-length check stands, so an accidental app/log mismatch
    still shows up as one.
    """
    out, incomplete = {}, {}
    for cluster, cores in sorted(per_core.items(), key=lambda kv: cluster_id(kv[0])):
        cid = cluster_id(cluster)
        table = app.table_for(cid)
        dm = find_dm_core(cores)

        # Validate the whole cluster before labelling any of it: a per-hart check
        # would leave a cluster half labelled and half raw, depending on the order
        # the harts happen to appear in the log.
        if table is not None:
            for hart, entries in cores.items():
                role = "dm" if hart == dm else "worker"
                want = len(table[role])
                if len(entries) == want:
                    continue
                # A short core is a truncated run; a long one cannot be this app.
                if partial_ok and len(entries) < want:
                    ran = max(len(e) for e in cores.values())
                    incomplete[cluster] = (ran, len(table["dm"]))
                    continue
                print(f"warning: {cluster} hart {hart} ran {len(entries)} "
                      f"regions, app has {want} for a {role} core "
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
                if labels is not None and idx < len(labels):
                    key = labels[idx]
                elif labels is not None:
                    key = f"[{role}] region {idx} (past the app's table)"
                else:
                    key = f"[{role}] region {idx}"
                r = regions.get(key)
                if r is None:
                    r = regions[key] = Region(key, start)
                r.add(cycles, start, end)
                r.order = min(r.order, start)
        out[cluster] = sorted(regions.values(), key=lambda r: r.order)
    return out, incomplete


def fmt_pct(x, width):
    """x is a ratio; prints as a percentage, or '-' when it is unknown."""
    return f"{'-':>{width}}" if x is None else f"{100.0 * x:>{width - 1}.1f}%"


# A DMA cannot beat its link.  A row that does is a region whose span is not the
# transfer's duration -- the engine reported done once the interconnect accepted
# the data, before it reached the far end.  Flagged rather than printed plain,
# because the number otherwise looks like a result.
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
    # OP/cyc prints whole (its fraction is below the resolution of anything it
    # is compared against); B/cyc keeps a decimal, since an off-chip peak need
    # not be a whole number of bytes per cycle.
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
        s = app.stage_of(cluster_id(cluster))
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
        # A global barrier delimits the step, so the window is the stage's, not
        # each cluster's own last region.  Lanes push concurrently over separate
        # links, hence one row of dout bytes each rather than one of their sum.
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

    # The label column is sized to its content, with the historical widths as the
    # floor: cevit labels fit in 46, celere's carry a whole layer description.
    lw = max([46] + [len(r.label) for rs in clusters.values() for r in rs])

    for cluster, regions in clusters.items():
        cid = cluster_id(cluster)
        stage = app.stage_of(cid)
        title = app.title_of(cid)
        busy = sum(r.span for r in regions)
        lo = min(r.start for r in regions)
        hi = max(r.end for r in regions)
        stage_span[cluster] = (lo, hi)

        print()
        print(f"=== {cluster}  (stage {stage}) -- {title}")
        print(f"    active {lo} -> {hi} cyc   sum of regions {busy} cyc")
        print(f"    {'region':<{lw}} {'cores':>5} {'span':>9} "
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
            print(f"    {r.label:<{lw}} {n:>5} {r.span:>9} {cells}"
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
    print(f"    {'kernel':<{lw - 2}}{'unit':>9} {'calls':>5} {'span':>8} "
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
        print(f"    {label:<{lw - 2}}{w.kind:>9} {calls:>5} {span:>8} "
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
            print(f"    {cluster:<10} {label:<{lw}} {rate:>7.1f} of {peak:g} B/cyc")
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
    for cluster, (lo, hi) in sorted(stage_span.items(),
                                    key=lambda kv: cluster_id(kv[0])):
        regions = clusters[cluster]
        busy = sum(r.span for r in regions)
        print(f"    {cluster}: window {hi - lo:>7} cyc, busy {busy:>7} cyc "
              f"({100.0 * busy / (hi - lo):5.1f}% of window)")

    # The lane clusters of a head-parallel stage run concurrently, so a stage
    # costs max(lane windows), not their sum.
    by_stage = defaultdict(list)
    for cluster, (lo, hi) in stage_span.items():
        cid = cluster_id(cluster)
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
            cid = cluster_id(cluster)
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
# One x slot per pipeline stage, the clusters of a head-parallel stage side by
# side inside a shaded slot.  Each region is charged by the unit work_model()
# gives it:
#
#   transfer -- DMA:NoC, DMA:LPDDR   (stage hand-offs and the HBM read/write)
#   compute  -- everything else: TE, PE and the DMA:L1 layout copies
#
# A lane's height is the SUM of its region spans, matching the CSV route number
# for number.  Each bar is subdivided into its regions, one segment per kernel
# and per DMA.  CTT orders segments bottom-to-top by execution; CWT splits them
# over two bars, grouped by side and ordered only within each.
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
    for cluster, regions in sorted(clusters.items(),
                                   key=lambda kv: cluster_id(kv[0])):
        cid = cluster_id(cluster)
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
        if cls in LAYER_COLORS:
            palette[cls] = LAYER_COLORS[cls]
    # Then the ones the palette has not met, each taking the first colour of its
    # side's pool nothing else on this figure uses.  Must run after every named
    # class has claimed its colour: the pools share entries with LAYER_COLORS
    # (RSHP and the first compute fallback are both #9467bd).
    for cls in order:
        if cls in palette:
            continue
        pool = LAYER_FALLBACK[side[cls]]
        free = [c for c in pool if c not in set(palette.values())]
        palette[cls] = free[0] if free else pool[fb[side[cls]] % len(pool)]
        fb[side[cls]] += 1
    # the stage input is an off-chip read, so in the kernel panel it is an LPDDR
    # segment like any other
    din_colour = palette.get("DMA LPDDR", "green")

    # Both panels share the y axis, so the subdivision below lines up row for row
    # with the split above.  Legends sit outside the axes.
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
                # CTT sums the two halves, so segment order does not change the
                # bar height: stack them in execution order.
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


def make_plots(clusters, app, args, incomplete=None):
    """Render whichever of --plot_ctt / --plot_cwt was asked for."""
    wanted = [(False, args.plot_ctt), (True, args.plot_cwt)]
    wanted = [(ov, path) for ov, path in wanted if path is not None]
    if not wanted:
        return
    # The test is for the ABSENCE of a stage layout, not for one particular
    # model: AppModel and CelereApp both provide one, NoApp is the case that
    # cannot be plotted.
    if isinstance(app, NoApp):
        print("error: --plot_ctt/--plot_cwt need --app (and not --raw): without the "
              "app there is no stage layout to plot", file=sys.stderr)
        return
    cycles_per_ms = args.freq_hz / 1e3
    stages = build_stages(clusters, app, cycles_per_ms)
    if not stages:
        print("error: no cluster mapped to a stage; nothing to plot", file=sys.stderr)
        return
    if incomplete:
        short = ", ".join(sorted(incomplete, key=cluster_id))
        print(f"warning: the log is partial, so {short} contribute only the regions "
              f"that ran; their stages are drawn shorter than they will be",
              file=sys.stderr)

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
               log.parent / f"{app.flavour}_bench_plots"
               / f"{app.plot_tag}_{tag}_split_{log.stem}.png")
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
    flavour = ap.add_mutually_exclusive_group()
    flavour.add_argument("--cevit", dest="flavour", action="store_const",
                         const="cevit",
                         help="the --app file is a cevit_pipeline.py app "
                              "(0-based stages, STAGE_DOUT_BYTES hand-offs)")
    flavour.add_argument("--celere", dest="flavour", action="store_const",
                         const="celere",
                         help="the --app file is a celere_pipeline.py app "
                              "(1-based stages, STAGE_XFER hand-offs, tiled last "
                              "stage).  Also tolerates a partial log, so a run "
                              "that hung or is still going still reports")
    ap.set_defaults(flavour=None)       # default: detect from the app source
    ap.add_argument("--partial", action="store_true",
                    help="label clusters that stopped part-way through their "
                         "region table instead of dropping them to raw "
                         "(implied by --celere)")
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
    ap.add_argument("--dram-peak", type=float, metavar="BPC",
                    help="peak bandwidth of ONE off-chip channel in B/cyc "
                         "(== GB/s at 1 GHz).  Default: read from the DRAMSys "
                         f"'MAX BW' footer in the log, else "
                         f"{DRAM_PEAK_FALLBACK_BPC:g}")
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
        seen = detect_flavour(args.app)
        if args.flavour is None:
            if seen is None:
                print(f"error: cannot tell which generator emitted {args.app} "
                      f"-- pass --cevit or --celere", file=sys.stderr)
                return 1
            args.flavour = seen
        elif seen is not None and seen != args.flavour:
            print(f"error: {args.app} looks like a {seen} app but --{args.flavour} "
                  f"was given -- pass --{seen}", file=sys.stderr)
            return 1
        app = (CelereApp if args.flavour == "celere" else AppModel)(args.app)
        print(f"app {args.app} [{args.flavour}]: {app.n_stages} stages, "
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
    dram_peaks = set()
    for line in open(args.log, errors="replace"):
        if "[Performance Counter]" in line:
            m = re.search(r"(\d+) ns", line)
            if m:
                total_ns = int(m.group(1))
        elif "MAX BW:" in line:
            m = DRAM_MAX_BW_RE.search(line)
            if m:
                dram_peaks.add(float(m.group(1)))

    set_dram_peak(args.dram_peak, dram_peaks)

    clusters, incomplete = build(per_core, app,
                                 partial_ok=args.partial or args.flavour == "celere")
    if incomplete:
        print()
        print("    NOTE: this log is partial -- the run did not reach the end of "
              "every stage.")
        print("    The clusters below are labelled as far as they got; their "
              "missing regions are")
        print("    absent from the totals, so treat those as lower bounds.")
        for cluster in sorted(incomplete, key=cluster_id):
            ran, of = incomplete[cluster]
            print(f"      {cluster:<12} reached region {ran} of {of}")
    # Only the serialized case pins the unbracketed hand-offs to a step of their
    # own; see add_handoff_regions().
    if getattr(app, "overlapped", False) and app.n_iter == 1:
        add_handoff_regions(clusters, app, total_ns)
    report(clusters, total_ns, app)
    if args.csv:
        write_csv(clusters, args.csv, app)
        print(f"\nwrote {args.csv}")
    make_plots(clusters, app, args, incomplete)
    return 0


if __name__ == "__main__":
    sys.exit(main())
