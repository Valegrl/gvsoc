#!/usr/bin/env python3
"""Aggregate the [BENCH] lines emitted by the mempool trace CSR (0x7d0) into a
per-kernel performance report.

Compute regions are scored against the op-issue peak of the unit that runs them
(TE, PE, or the LSU for the arithmetic-free patch permutations) and DMA regions
against their link's peak bandwidth (off-chip LPDDR, inter-cluster NoC,
cluster-local), so every row carries a utilization.

Every core writes CSR_TRACE around each region, so a run produces one line per
(core, region).  This groups them back into the layers of the pipeline.

    ./parse_bench.py log_bench.log --app main.c
    ./parse_bench.py log_bench.log --app main.c --csv out.csv
    ./parse_bench.py log_bench.log --raw        # no labels, just region indices
    ./parse_bench.py log_bench.log --app main.c --plot_ctt   # and the stage figure
    ./parse_bench.py celere.log --app celere.c --celere      # the celere generator

A [BENCH] record identifies its region only by a numeric index, so the kernel
names, transfer sizes and GEMM shapes are read out of the app source that
produced the log.  Without --app the run is reported raw, by region index --
guessing a layout would silently produce wrong utilizations rather than none.

Two generators emit apps this reads, and they lay out their stages differently:
AppModel covers cevit_pipeline.py and CelereApp celere_pipeline.py, each tracking
the CURRENT generator only.  --cevit / --celere pick a model; without either the
flavour is detected from the tables the app source declares.  --celere also
tolerates a PARTIAL log (a run that hung, was killed, or is still going): those
clusters are labelled as far as they got, so their totals are lower bounds.
"""

import argparse
import csv
import math
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
    """-> the N of 'cluster_N'; also the sort key wherever clusters are listed,
    since the names sort as strings (cluster_10 ahead of cluster_2)."""
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
# The heading both generators put on a compute function; they space the colon
# differently: '/* stage 2 : [ZN : Normalize on dim i_4] */' (celere),
# '/* stage 2: rows 14, 15, 16, 8 lanes over i_2 x i_3 */' (cevit).
STAGE_TITLE_RE = re.compile(r"^/\* stage (\d+)\s*:\s*(.*?)\s*\*/", re.M)
# The compute-while-transfer generator passes the item parity in, so the
# signature is compute_stage_N(uint32_t buf) there and compute_stage_N(void) in
# the blocking one.  THE QUALIFIERS BETWEEN `static` AND `void` ARE NOT FIXED:
# the generator marks the stage bodies NO_INLINE (its own macro for
# __attribute__((noinline))) to give each one its own stack frame, and used to
# mark them `inline`.  Anything of that shape is accepted, since a missed match
# here costs the whole app its kernel labels and reports raw regions.
COMPUTE_FN_RE = re.compile(
    r"^static[ \t]+(?:\w+[ \t]+)*?void[ \t]+compute_stage_(\d+)\s*\([^)]*\)"
    r"\s*\n\{(.*?)^\}", re.M | re.S)
# The DMA helpers the generator emits, newest spelling first.  Each brackets
# itself, so every CALL is a region.  dma_1d() is the older generator's
# contiguous helper, kept so its logs still parse.
DMA_CALLS = ("dma_2d_strided", "dma_2d", "dma_1d")
# The bracket a measured region sits in.  A REGION IS ONE start/stop PAIR, NOT
# ONE KERNEL CALL, and the two stopped agreeing once the layout rows became real
# kernels: the generator brackets all the pieces of one row together (a fold
# split into whole windows and a remainder is two calls, one region) and
# brackets a pure VIEW -- a row whose operand already sits in the order its
# consumer wants -- around nothing at all.  Counting calls gives the first case
# too many regions and the second none, and either one slides the rest of the
# cluster's table against the log and costs it every label.
BENCH_START_RE = re.compile(r"^mempool_start_benchmark\s*\(")
BENCH_STOP_RE = re.compile(r"^mempool_stop_benchmark\s*\(")
# 'RSHP QF, KF : Fold ... 572x128' -> ('RSHP', 'QF, KF : Fold ...', 572, 128).
# The shape is what _fold_region() adds up when a bracket holds several calls.
SHAPE_LABEL_RE = re.compile(r"^(\w[\w&+/.-]*)\s+(.*?)\s+(\d+)x(\d+)$")
# "Skip Sum [add]: Y += X" -> ("Skip Sum", "add").  The tag is one word but not
# always one \w+ run (celere spells a fused layout op '[split&reshape]'); a layer
# heading ('FFN[rows 1-3] x1') has a space in its bracket and so is not a region.
COMMENT_NAME_RE = re.compile(r"^(.*?)\s*\[([\w&+/.-]+)\]")
# Matches the whole off-chip helper family: hbm_addr() for the default edge,
# hbm_<edge>() for a specific one.  A call using any of them is an LPDDR leg,
# not a cluster-local layout copy.
HBM_CALL_RE = re.compile(r"\bhbm_\w*\s*\(")
# Where the app says which stage runs which hand-off, in both spellings the
# generator has used: `case STAGE_1: push_qkv(...)` and
# `if (stage == STAGE_1) push_qkv(...)`.  The stage number is an enum member, so
# it is read off the name.  This matches the COMPUTE switch too; the callee is
# checked for a DMA of its own (in _push_sources) rather than for a name that
# looks like a push, so the compute arms drop out on their own.
PUSH_DISPATCH_RE = re.compile(
    r"(?:case\s+STAGE_(\d+)\s*:|stage\s*==\s*STAGE_(\d+)\s*\))\s*(\w+)\s*\(")
# main()'s own statement of the schedule -- `phase = stage - 1u` for ctt,
# `phase = 2u * (stage - 1u)` for cwt.  The STRIDE between two stages' phases
# tells the two apart, and is the same factor the latency is quoted in
# (steps = stride x stages).
PHASE_RE = re.compile(r"\bphase\s*=\s*([^;]+);")
# The consumer-side region offset in a hand-off destination, which names the
# tensor that leg carries: 'y_base + FFN_RESID_Y + ...' -> 'FFN_RESID'.  The
# generator suffixes the offset with the buffer it indexes (_X inbound, _Y
# outbound).  Non-greedy, so QKV_Q_X is QKV_Q, not QKV.
PUSH_DEST_OFF_RE = re.compile(r"\b(\w+?)_(?:OFF|[XY])\b")
# `static const uint32_t DST[3] = { QKV_Q_X, QKV_K_X, QKV_V_X };` -- a hand-off
# that walks the q|k|v stripes picks its destination out of a table indexed by
# the stripe loop, so the tensor a descriptor carries is named by the ENTRY and
# appears nowhere in the DMA call.  Kept as unexpanded text: the point of the row
# is the macro's name.
CONST_TABLE_RE = re.compile(
    r"^static\s+const\s+u?int(?:8|16|32|64)_t\s+(\w+)\s*\[[^\]]*\]"
    r"\s*=\s*\{([^}]*)\}\s*;$")
# Where a resolved const table is parked in a hand-off's environment.  Not a
# valid C identifier, so it cannot collide with one of the app's own locals.
TABLES_KEY = "@tables"
# The lpddr_t declaration, and one field of it.  The rows of LPDDR_IN[] /
# LPDDR_OUT[] are read by field NAME rather than by position: the generator
# carries whatever columns the placement needs, and the struct says which of them
# is `bytes`.
LPDDR_STRUCT_RE = re.compile(r"typedef\s+struct\s*\{(.*?)\}\s*lpddr_t\s*;", re.S)
STRUCT_FIELD_RE = re.compile(r"\b\w+\s+(\w+)\s*;")
# The statement shapes a push_*() helper is built from.  Only the counted `for`
# (`i = A; i < B; i++`) is supported, since a loop this evaluator cannot count is
# a wrong region count, not just a missing label.
FOR_RE = re.compile(r"^for\s*\(\s*(?:\w[\w\s*]*?\s+)?(\w+)\s*=\s*([^;]+);\s*"
                    r"\1\s*<\s*([^;]+);\s*\1\s*\+\+\s*\)\s*\{?\s*$")
IF_RE = re.compile(r"^if\s*\((.*)\)\s*\{?\s*$")
# The generator writes an empty overlap as an early exit (`if (lo >= hi)
# continue;`), so a jump the evaluator ignored would issue exactly the
# descriptors the app skips.
JUMP_RE = re.compile(r"^(continue|break)\s*;$")
# An integer declaration, which may declare several names at once:
# `uint32_t g = c / QSPLIT, q = c % QSPLIT;`.  The declarator list is split at
# top level afterwards, since a comma also appears inside an initialiser.
DECL_RE = re.compile(r"^(?:const\s+)?u?int(?:8|16|32|64)_t\s+(.+);$")
# A runaway trip count means the bound was misread; stop rather than build a
# region table of millions of rows.
MAX_TRIPS = 100000

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
    # patchify_f16p(src, dst, tok0, f0, n_tok, Nt, Nh, Nw, C, cid, nt) and its
    # inverse.  The first four arguments are per-lane expressions, not literals
    # (`lane * 546u`), so only the shape is read.  The label carries the shape
    # the kernel WALKS -- n_tok*Nh units of Nw -- since that is what layout_work()
    # costs; see reshape_label().
    (re.compile(rf"(un)?patchify_f16p\(\s*{B},\s*{B},\s*{B},\s*{B},\s*"
                r"(\d+)u?,\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?"),
     lambda n, g, t: reshape_label(n, g)),
    # transpose_f16p(src, dst, rows, cols, cid, nt).  Layout work like the
    # reshape above and modelled the same way, but its own legend class: it is
    # one of the model's own rows (transpose K), not the grid packing.
    (re.compile(rf"transpose_f16p\(\s*{B},\s*{B},\s*(\d+)u?,\s*(\d+)u?"),
     lambda n, g, t: f"TRSP {n} {g[0]}x{g[1]}"),
    # mempool_wait(3805UL) -- a bracketed stand-in for a kernel that is not in
    # this tree (cevit's unridden layout rows, celere's FFTs and rope).  One call
    # covers all of them, so the label leads with the comment's own [tag] to keep
    # them as separate legend classes.
    (re.compile(r"mempool_wait\(\s*(\d+)[uUlL]*\s*\)"),
     lambda n, g, t: f"{_stand_in_label(n, t)} ({g[0]} cyc stand-in)"),
]


def _stand_in_label(name, tag):
    """-> 'SPLIT split q,k,v' / 'FFT on dim i_2' for a mempool_wait region.

    region_class() reads the class off the label's FIRST word, so the tag leads
    -- except where the name already opens with it ('FFT on dim i_2 [fft]'),
    where that word is raised to the class's spelling instead of being repeated.
    'IFFT' is deliberately not treated as already-led: prefixing is what puts an
    inverse transform in the FFT class.
    """
    cls = (tag or "wait").upper()
    parts = name.split(None, 1)
    if parts and parts[0].upper() == cls:
        return " ".join([cls] + parts[1:])
    return f"{cls} {name}"


def reshape_label(name, g):
    """-> the label for a patchify_f16p / unpatchify_f16p call.

    g is ('un' or None, n_tok, Nt, Nh, Nw, C).  A patch is Nh x Nw, so this
    lane's slab is n_tok tokens of Nh*Nw elements -- but the kernel deals out a
    finer unit than a token: a WORK UNIT is one PATCH ROW, u = j*Nh + a, and a
    core takes a contiguous chunk of the n_tok*Nh of them.  The label carries
    that unit shape, because it is what layout_work() has to divide over the
    cores; the element count is the same either way, so both directions cost the
    same and share one model.
    """
    n_tok, nt, nh, nw, c = (int(x) for x in g[1:])
    return (f"RSHP {name} {n_tok * nh}x{nw} "
            f"({'unpatchify' if g[0] else 'patchify'})")


def _comment_parts(text):
    """-> (name, tag) of a trailing comment: 'Skip Sum [add]: ...' -> ('Skip Sum', 'add').

    The tag is the generator's own name for the kernel class -- not the C
    function implementing it -- and is what tells the mempool_wait() stand-ins
    apart.  tag is None when the comment carries no bracket.
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

    dma_1d/dma_2d carry the count as their third argument; the strided form moves
    `repeat` chunks of `size` (how a transpose is issued).
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


# Both generators lay their geometry out in tables of structs and put the
# tensor's name in each row's trailing comment, so the reader is shared:
#   `{ 1171456u,   1u,  536064u },  /* stage 1: local-pass K */`
# The fields are left to the caller (each table has its own).  Deliberately not
# DOTALL: a row and its comment are one line.
TABLE_ROW_RE = re.compile(r"\{([^}]*)\}[^\n/]*(?:/\*\s*(.*?)\s*\*/)?")
# Every row comment opens with the stage it belongs to ('stage 1: local-pass Q'),
# which the label already carries.
ROW_STAGE_RE = re.compile(r"^\s*stage\s+\d+\s*:\s*", re.I)


def table_rows(src, name):
    """-> [(fields, comment), ...] for the initialiser of `name[]`, or None.

    The declaration is matched, never a mention -- the generators name their
    tables in the comments above them, so `[...] = {` is what tells the two
    apart.  Fields come back as raw strings, since each table spells its own.
    """
    m = re.search(rf"\b{name}\s*\[[^\]\n]*\]\s*=\s*\{{(.*?)\n\}}", src, re.S)
    if not m:
        return None
    out = []
    for fields, comment in TABLE_ROW_RE.findall(m.group(1)):
        out.append(([f.strip() for f in fields.split(",") if f.strip()],
                    ROW_STAGE_RE.sub("", comment or "").strip()))
    return out


def expand_defines(src, d):
    """Substitute the integer #defines so the size and trip-count regexes see literals."""
    if not d:
        return src
    rx = re.compile(r"\b(" + "|".join(map(re.escape, sorted(d, key=len,
                                                            reverse=True)))
                    + r")\b")
    return rx.sub(lambda m: str(d[m.group(1)]), src)


# --------------------------------------------------------------------------
# A very small C integer-expression evaluator.
#
# The cevit hand-off is a loop nest over index arithmetic, and BOTH the region
# count and the transfer sizes fall out of it: `reps` is `hi - lo`, and a
# `if (lo < hi)` decides whether a transfer happens at all.  Neither survives
# pattern matching, so the nest is evaluated per lane.
#
# Only what the generator emits is read: unsigned integer arithmetic, the
# comparisons, the ternary, indexing a const table, and calls to the app's own
# one-line `static inline` helpers (lo_of/hi_of).  Anything else evaluates to
# None, which the caller turns into either an unsized label or a hard error,
# depending on whether a region COUNT depended on it.
# --------------------------------------------------------------------------

CAST_RE = re.compile(
    r"\(\s*(?:u?int(?:8|16|32|64)_t|uintptr_t|size_t|unsigned(?:\s+\w+)?|int)\s*\)")
INT_SUFFIX_RE = re.compile(r"\b(\d+)[uUlL]+")
# C's `/` is integer division on these operands, so it becomes Python's `//`.
# The lookarounds keep an existing `//` from doubling up.
DIV_RE = re.compile(r"(?<!/)/(?!/)")
# The app's own one-liners: `static inline uint32_t lo_of(uint32_t a, uint32_t b)
# { return a > b ? a : b; }`.  Inlined rather than special-cased by name, so a
# generator that renames or adds one keeps working.
INLINE_FN_RE = re.compile(
    r"^\s*static\s+inline\s+\w+\s+(\w+)\s*\(([^)]*)\)\s*\{\s*return\s+([^;]+);\s*\}",
    re.M)
ENUM_RE = re.compile(r"\benum\s*\{([^}]*)\}\s*;")


def _ternary_to_python(e):
    """'a > b ? a : b' -> '((a) if (a > b) else (b))', at every nesting depth."""
    depth = 0
    for i, ch in enumerate(e):
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == "?" and depth == 0:
            j, d2, nest = i + 1, 0, 0
            while j < len(e):
                c = e[j]
                if c == "(":
                    d2 += 1
                elif c == ")":
                    d2 -= 1
                elif d2 == 0 and c == "?":
                    nest += 1
                elif d2 == 0 and c == ":":
                    if not nest:
                        break
                    nest -= 1
                j += 1
            if j >= len(e):
                break                           # unbalanced; let eval() fail
            return (f"(({_ternary_to_python(e[i + 1:j])})"
                    f" if ({_ternary_to_python(e[:i])})"
                    f" else ({_ternary_to_python(e[j + 1:])}))")
    return e


def _to_python(expr):
    """-> the Python spelling of a C integer expression."""
    e = COMMENT_RE.sub(" ", expr)
    e = CAST_RE.sub(" ", e)
    e = INT_SUFFIX_RE.sub(r"\1", e)
    e = DIV_RE.sub("//", e)
    e = e.replace("&&", " and ").replace("||", " or ")
    e = e.replace("!=", "\0").replace("!", " not ").replace("\0", "!=")
    return _ternary_to_python(e)


def c_eval(expr, env):
    """-> the value of a C expression under `env`, or None if it is not one we read.

    None propagates: a name bound to None makes every expression using it None
    too, so a local this evaluator could not work out never silently becomes a
    number somewhere downstream.
    """
    try:
        return eval(_to_python(expr), {"__builtins__": {}}, env)   # noqa: S307
    except Exception:
        return None


def _inline_fn(params, body, env):
    """-> a callable for one `static inline` one-liner, evaluated at call time."""
    return lambda *a: c_eval(body, {**env, **dict(zip(params, a))})


# One off-chip leg of a cevit app: one row of LPDDR_IN[] / LPDDR_OUT[].
# lpddr_move() walks the stage's slice of the table and loops over `pieces`
# inside each entry, so `nbytes` is the size of ONE piece and `pieces` is the
# region count.  `edge` rides along because the direction does not imply it:
# stage 3 reads its residual off a SOUTH node and drains north.
CevitLeg = namedtuple("CevitLeg", "nbytes edge pieces name")


class _Continue(Exception):
    """A `continue` in a hand-off nest, unwinding to the innermost loop.

    The nest is walked by recursive descent, so a jump has to cross however many
    `if` bodies sit between it and its loop -- hence an exception rather than a
    return code threaded through every level.
    """


class _Break(Exception):
    """A `break` in a hand-off nest, leaving the innermost loop."""


class AppModel:
    """The region layout of one generated cevit app, read from its main.c.

    Tracks the CURRENT cevit_pipeline.py, whose apps number their stages
    1..N_STAGES (slot 0 of every table is dead) and describe themselves with:

      * CID_STAGE / CID_LANE / STAGE_BASE -- the placement.  There is no
        STAGE_LANES table: a stage's width is how many cids CID_STAGE gives it;
      * LPDDR_IN[] / LPDDR_OUT[], rows of lpddr_t sliced per stage by
        LPDDR_IN_FIRST / LPDDR_OUT_FIRST, one bracketed dma_2d each;
      * push_to_next(), which dispatches on the stage to a push_*() helper whose
        LOOP NEST is the hand-off -- there is no table of transfer descriptors,
        so the helper's own for-bounds and DMA arguments are what is read.
    """

    flavour = "cevit"           # names the default plot directory and file stem
    plot_tag = "CEViT"

    def __init__(self, path):
        src = Path(path).read_text()
        bare = COMMENT_RE.sub(" ", src)     # comments carry digits of their own

        d = {m.group(1): int(m.group(2)) for m in DEFINE_RE.finditer(bare)}
        self.n_stages = d["N_STAGES"]
        self.n_iter = d.get("N_ITER", 1)

        self.cid_stage = self._array(bare, "CID_STAGE")
        self.cid_lane = self._array(bare, "CID_LANE")
        self.stage_base = self._array(bare, "STAGE_BASE")
        # Lane counts are COUNTED, not read: CID_STAGE already says which stage
        # owns each cid, and one cid is one lane.  Slot 0 stays dead -- cid_stage
        # 0 marks an IDLE cluster, not a stage-0 lane.
        self.lanes = [self.cid_stage.count(sid)
                      for sid in range(self.n_stages + 1)]
        self.lanes[0] = 0
        self.in_first = self._array(bare, "LPDDR_IN_FIRST")
        self.out_first = self._array(bare, "LPDDR_OUT_FIRST")
        # The struct tables are read out of the SOURCE, not `bare`: each row's
        # trailing comment names the tensor it carries, and that name is what
        # makes an off-chip row readable in the report.
        self.lpddr_in = self._lpddr_table(src, "LPDDR_IN")
        self.lpddr_out = self._lpddr_table(src, "LPDDR_OUT")

        # Which schedule this app runs, read off main()'s own phase expression:
        # ctt puts stage s on item m at step m + s, cwt at step m + 2s, leaving a
        # spare step between a producer and its consumer for the transfer.  Both
        # bracket every DMA, so this changes no region's existence.
        self.overlapped = self._phase_stride(bare) > 1

        # The environment the hand-off nests are evaluated in: the #defines, the
        # stage enum, the placement tables, and the app's own one-line helpers.
        self.env = dict(d)
        self.env.update(self._enums(bare))
        self.env.update(CID_STAGE=self.cid_stage, CID_LANE=self.cid_lane,
                        STAGE_BASE=self.stage_base, STAGE_LANES=self.lanes,
                        LPDDR_IN_FIRST=self.in_first,
                        LPDDR_OUT_FIRST=self.out_first)
        for name, params, body in INLINE_FN_RE.findall(src):
            names = [p.split()[-1].lstrip("*")
                     for p in params.split(",") if p.strip()]
            self.env[name] = _inline_fn(names, body, self.env)
        self.push_src = self._push_sources(src)

        self.titles = {int(m.group(1)): m.group(2)
                       for m in STAGE_TITLE_RE.finditer(src)}
        self.bodies = {int(m.group(1)): self._body(m.group(2))
                       for m in COMPUTE_FN_RE.finditer(src)}

        self.stages = {sid: {"name": self.titles.get(sid, "?")}
                       for sid in range(1, self.n_stages + 1)}
        # The region table is per CLUSTER, not per stage: two lanes of one stage
        # push different amounts, and sometimes a different NUMBER of times
        # (a transfer whose token overlap is empty is not issued at all).
        self.tables = {}
        for cid, sid in enumerate(self.cid_stage):
            if sid:                             # 0 marks a cid left idle
                dm, worker = self._cluster_regions(cid, sid)
                self.tables[cid] = {"dm": dm, "worker": worker}

    @staticmethod
    def _array(bare, name):
        m = re.search(rf"{name}\s*\[[^\]]*\]\s*=\s*\{{([^}}]*)\}}", bare)
        if not m:
            raise ValueError(f"no {name}[] in app source")
        return [int(x) for x in re.findall(r"(\d+)", m.group(1))]

    @staticmethod
    def _phase_stride(bare):
        """-> the steps between one stage's phase and the next's: 1 ctt, 2 cwt.

        Evaluated from main()'s own expression at two stages rather than pattern
        matched, so a different spelling of the same schedule still reads right.
        """
        m = PHASE_RE.search(bare)
        if not m:
            raise ValueError("no `phase = ...` in app source: cannot tell a "
                             "compute-then-transfer app from a "
                             "compute-while-transfer one")
        first = c_eval(m.group(1), {"stage": 1})
        second = c_eval(m.group(1), {"stage": 2})
        if first is None or second is None or second <= first:
            raise ValueError(f"cannot read the schedule from "
                             f"`phase = {m.group(1).strip()}`")
        return second - first

    @staticmethod
    def _logical_lines(text):
        """-> the body, one C statement per entry.

        A call may be split over several source lines (the strided DMA is), so
        lines are joined until their parentheses balance again.  Comments are
        kept -- they name the region -- but ignored while counting.
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
    def _fold_region(labels, ctx, dm):
        """-> [(label, dm_only)] for the ONE region a start/stop bracket measures.

        Three shapes reach this, in the order the generator emits them:

          one call    the common row; its own label, unchanged
          several     one layer row cut into pieces (a fold over whole windows
                      plus a remainder, a gather of Q then K then V).  The pieces
                      are the same kernel over the same unit width, so the region
                      costs their SUM -- summing the unit counts keeps the label
                      in the shape work_model() reads and scores the region as
                      the one span the log timed
          none        a VIEW: the row is a relabelling its consumer reads as-is,
                      so the generator brackets the comment alone and the log
                      carries a region of about one cycle.  It still has to
                      appear, or every later region in the stage shifts by one

        A fold of pieces this cannot add up (different kernels, or a shape that
        is not `UxV`) keeps every piece in the label and goes unscored, which
        loses the row's utilization but never its place in the sequence.
        """
        if not labels:
            text = ctx["comment"]
            ctx["comment"] = None
            name, tag = _comment_parts(text) if text else ("view", None)
            return [(f"{_stand_in_label(name, tag)} (view, no work)", dm)]
        if len(labels) == 1:
            return labels
        dm_only = all(d for _, d in labels)
        parts = [SHAPE_LABEL_RE.match(label) for label, _ in labels]
        if all(parts) and len({(p.group(1), p.group(4)) for p in parts}) == 1:
            head = parts[0]
            total = sum(int(p.group(3)) for p in parts)
            return [(f"{head.group(1)} {head.group(2)} "
                     f"{total}x{head.group(4)}", dm_only)]
        return [(" + ".join(label for label, _ in labels), dm_only)]

    @staticmethod
    def _scan(lines, i, dm, ctx, specs=None):
        """-> ([(label, dm_only), ...], index past the block that starts at i).

        Walks the block structure rather than the lines: a region inside a `for`
        is that many regions in the log, and a `if (flex_is_dm_core())` block
        makes every region inside it dm-only -- which a per-line test would miss
        when the guard sits on a line of its own.

        Between a mempool_start_benchmark() and its stop, what the statements
        yield is held back and folded into the single region the pair measures;
        outside one, a statement that brackets ITSELF (every DMA helper does) is
        a region on its own, as before.
        """
        out, pend = [], None
        while i < len(lines):
            line = lines[i]
            i += 1
            bare = COMMENT_RE.sub(" ", line).strip()
            if bare.startswith("}"):
                break
            if not bare:                        # a comment and nothing else
                m = re.search(r"/\*\s*(.*?)\s*\*/", line)
                # Only a region comment ('name [tag]: ...') labels a region; the
                # layer headings the generator emits ('FFN[rows 1-3] x1') do not.
                if m and COMMENT_NAME_RE.match(m.group(1)):
                    ctx["comment"] = m.group(1)
                continue
            if BENCH_START_RE.match(bare):
                pend = []
                continue
            if BENCH_STOP_RE.match(bare):
                out.extend(AppModel._fold_region(pend or [], ctx, dm))
                pend = None
                continue
            trips = 1
            if bare.startswith("for"):
                m = re.search(r"<\s*(\d+)", bare)
                trips = int(m.group(1)) if m else 1
            # the guard covers this statement or block only, never its siblings
            dm_here = dm or "flex_is_dm_core" in bare
            sink = out if pend is None else pend
            if bare.endswith("{"):
                body, i = AppModel._scan(lines, i, dm_here, ctx, specs)
                sink.extend(body * trips)
                continue
            sink.extend(AppModel._region_of(line, ctx, dm_here, specs))
        if pend:                        # an unclosed bracket: keep its regions
            out.extend(pend)
        return out, i

    @staticmethod
    def _body(text):
        """-> [(label, dm_only), ...] for the regions of one compute_stage_N()."""
        regions, _ = AppModel._scan(AppModel._logical_lines(text), 0, False,
                                    {"comment": None})
        return regions

    # -- source tables ----------------------------------------------------

    @staticmethod
    def _lpddr_table(src, name):
        """-> [CevitLeg, ...], one entry per lpddr_t row of `name[]`.

        The placement columns of a row come and go with the generator, so the row
        is matched against the app's OWN lpddr_t declaration and read by field
        name: only `edge`, `bytes` and `pieces` are used, and reading them by
        position is what a new column would silently break.  A declaration with
        no `pieces` cuts nothing up, so its rows move one piece each.
        """
        decl = LPDDR_STRUCT_RE.search(COMMENT_RE.sub(" ", src))
        if not decl:
            raise ValueError("no lpddr_t declaration in app source")
        names = STRUCT_FIELD_RE.findall(decl.group(1))
        rows = table_rows(src, name)
        if rows is None:
            raise ValueError(f"no {name}[] in app source")
        out = []
        for fields, comment in rows:
            vals = [_uint(f) for f in fields]
            if len(vals) != len(names) or any(v is None for v in vals):
                raise ValueError(
                    f"{name}[] row {{{', '.join(fields)}}} does not fit the "
                    f"app's own lpddr_t ({{ {', '.join(names)} }})")
            leg = dict(zip(names, vals))
            out.append(CevitLeg(leg["bytes"],
                                "north" if leg["edge"] else "south",
                                leg.get("pieces", 1), comment))
        return out

    @staticmethod
    def _enums(bare):
        """-> {name: value} for the app's enum members ('enum { STAGE_1 = 1, ... }').

        The stage numbers reach push_to_next() and STAGE_BASE[] as these names,
        not as literals, so the tables cannot be indexed without them.
        """
        out = {}
        for block in ENUM_RE.findall(bare):
            nxt = 0
            for item in block.split(","):
                item = item.strip()
                if not item:
                    continue
                if "=" in item:
                    name, val = item.split("=", 1)
                    name, nxt = name.strip(), int(val.strip().rstrip("uUlL"))
                else:
                    name = item
                out[name] = nxt
                nxt += 1
        return out

    @staticmethod
    def _push_sources(src):
        """-> {stage: [logical lines]} for the app's hand-off helpers.

        The dispatch says which stages push at all -- one missing from it (the
        last, and any that drains itself) has no entry.  A candidate is kept only
        if the function it names issues a DMA, which separates the transfer arms
        from the compute arms in the same switch.

        The nest is kept as SOURCE and evaluated once per lane: its trip counts
        and guards are what differ between the lanes of one stage.
        """
        out = {}
        for case_sid, if_sid, fn in PUSH_DISPATCH_RE.findall(
                COMMENT_RE.sub(" ", src)):
            nest = AppModel._fn_body(src, fn)
            if nest is None or not any(call_args(nest, d) for d in DMA_CALLS):
                continue                        # a compute arm, or not a call
            out[int(case_sid or if_sid)] = AppModel._logical_lines(nest)
        if not out:
            raise ValueError("no stage -> hand-off dispatch in app source: "
                             "nothing of the form `case STAGE_n: push(...)`, "
                             "and no push_to_next()")
        return out

    @staticmethod
    def _fn_body(src, name):
        """-> the body of `static inline ... name(...) { ... }`, or None.

        Brace-counted rather than matched to a column-0 '}', since the push
        helpers nest their loops and the compute functions do not.
        """
        m = re.search(rf"^\s*static\s+(?:inline\s+)?\w[\w\s*]*\b{name}\s*\("
                      rf"[^)]*\)\s*\{{", src, re.M | re.S)
        if not m:
            return None
        depth, i = 1, m.end()
        while i < len(src) and depth:
            depth += (src[i] == "{") - (src[i] == "}")
            i += 1
        return src[m.end():i - 1]

    # -- executing a hand-off nest ----------------------------------------

    @staticmethod
    def _brace_body(lines, i):
        """-> (the lines inside a block whose '{' closed line i-1, index past '}')."""
        depth, j = 1, i
        while j < len(lines) and depth:
            bare = COMMENT_RE.sub(" ", lines[j]).strip()
            depth += bare.count("{") - bare.count("}")
            j += 1
        return lines[i:j - 1], j

    @staticmethod
    def _one_stmt(lines, i):
        """-> (the single statement at i, index past it).

        A braceless `for` or `if` governs ONE statement, which may itself be a
        `for` or an `if` -- so the statement is however many lines that takes,
        not one line.
        """
        while i < len(lines) and not COMMENT_RE.sub(" ", lines[i]).strip():
            i += 1
        if i >= len(lines):
            return [], i
        bare = COMMENT_RE.sub(" ", lines[i]).strip()
        start, i = i, i + 1
        if bare.endswith("{"):
            _, i = AppModel._brace_body(lines, i)
        elif re.match(r"(for|if|while)\s*\(", bare):
            _, i = AppModel._one_stmt(lines, i)
        return lines[start:i], i

    @staticmethod
    def _sub_body(lines, i, header):
        """-> (the body a `for`/`if` header governs, index past it)."""
        if header.endswith("{"):
            return AppModel._brace_body(lines, i)
        return AppModel._one_stmt(lines, i)

    @staticmethod
    def _exec(lines, env, out, where):
        """Run one push_*() body under `env`, appending (nbytes, tensor) per DMA.

        Both the sizes AND the region count are results of running the nest, so a
        bound, a guard or a jump this cannot follow is an error -- it would
        silently give the cluster the wrong number of regions -- while a `reps`
        it cannot evaluate only costs the row its size.
        """
        i = 0
        while i < len(lines):
            line = lines[i]
            bare = COMMENT_RE.sub(" ", line).strip()
            i += 1
            if not bare or bare in ("{", "}"):
                continue
            if bare.startswith("for"):
                m = FOR_RE.match(bare)
                if not m:
                    raise ValueError(f"{where}: cannot count the loop `{bare}`")
                body, i = AppModel._sub_body(lines, i, bare)
                lo, hi = c_eval(m.group(2), env), c_eval(m.group(3), env)
                if lo is None or hi is None:
                    raise ValueError(f"{where}: cannot evaluate the bounds of "
                                     f"`{bare}`")
                if hi - lo > MAX_TRIPS:
                    raise ValueError(f"{where}: `{bare}` runs {hi - lo} times, "
                                     f"which is not a region count")
                for v in range(lo, hi):
                    env[m.group(1)] = v
                    try:
                        AppModel._exec(body, env, out, where)
                    except _Continue:
                        continue
                    except _Break:
                        break
                continue
            if bare.startswith("if"):
                m = IF_RE.match(bare)
                if not m:
                    raise ValueError(f"{where}: cannot read the guard `{bare}`")
                body, i = AppModel._sub_body(lines, i, bare)
                cond = c_eval(m.group(1), env)
                if cond is None:
                    raise ValueError(f"{where}: cannot evaluate the guard "
                                     f"`{m.group(1)}`")
                if cond:
                    AppModel._exec(body, env, out, where)
                continue
            m = JUMP_RE.match(bare)
            if m:
                raise _Continue() if m.group(1) == "continue" else _Break()
            m = CONST_TABLE_RE.match(bare)
            if m:
                entries = [e.strip() for e in m.group(2).split(",") if e.strip()]
                # Both spellings: the VALUES so an index of the table evaluates
                # as an address, the ENTRY TEXT so the descriptor can be named
                # after the region it lands in.
                env[m.group(1)] = [c_eval(e, env) for e in entries]
                env.setdefault(TABLES_KEY, {})[m.group(1)] = entries
                continue
            m = DECL_RE.match(bare)
            if m:
                for name, init in AppModel._declarators(m.group(1)):
                    # None when the initialiser is an address rather than an
                    # index; c_eval() propagates it, so nothing downstream reads
                    # it back as a 0.
                    env[name] = None if init is None else c_eval(init, env)
                continue
            out.extend(AppModel._push_dma(line, env))
        return out

    @staticmethod
    def _declarators(text):
        """-> [(name, initialiser or None), ...] for one declaration's list.

        Split at TOP-LEVEL commas only: `f(a, b)` in an initialiser is one
        declarator, not two.
        """
        parts, cur, depth = [], "", 0
        for ch in text:
            if ch in "([":
                depth += 1
            elif ch in ")]":
                depth -= 1
            if ch == "," and not depth:
                parts.append(cur)
                cur = ""
            else:
                cur += ch
        out = []
        for part in parts + [cur]:
            name, _, init = part.partition("=")
            name = name.strip().lstrip("*")
            if name:
                out.append((name, init.strip() or None))
        return out

    @staticmethod
    def _dest_tensor(dest, env):
        """-> the tensor a hand-off descriptor lands in, or None.

        The consumer-side REGION OFFSET is the only thing telling one leg of a
        hand-off from another once their sizes are equal (K and V cross at the
        same size).  It reaches the address written out
        (`y_base + FFN_RESID_Y + ...`) or through a const table indexed by the
        stripe loop (`x_base + DST[s] + ...`); an indexed table is resolved to
        the entry THIS trip names first.
        """
        for name, entries in env.get(TABLES_KEY, {}).items():
            m = re.search(rf"\b{re.escape(name)}\s*\[([^\]]*)\]", dest)
            if not m:
                continue
            k = c_eval(m.group(1), env)
            if k is None or not 0 <= k < len(entries):
                return None
            dest = entries[k]
            break
        m = PUSH_DEST_OFF_RE.search(dest)
        return m.group(1) if m else None

    @staticmethod
    def _push_dma(line, env):
        """-> [(nbytes, tensor)] if this statement is a hand-off DMA, else []."""
        for fn in DMA_CALLS:
            args = call_args(line, fn)
            if args is None:
                continue
            idx = (2, 5) if fn == "dma_2d_strided" else (2,)
            nbytes = 1
            for k in idx:
                v = c_eval(args[k], env) if k < len(args) else None
                nbytes = None if v is None or nbytes is None else nbytes * v
            return [(nbytes, AppModel._dest_tensor(args[0], env))]
        return []

    # -- region tables ----------------------------------------------------

    def _lpddr_regions(self, sid, table, first, arrow, what):
        """The bracketed DMAs lpddr_move() issues for stage sid.

        One per PIECE of every row in the stage's range: a cluster owning two
        pieces of a cut tensor issues two transfers of one piece each, not one of
        their sum.  None at all for a stage that touches no off-chip node.
        """
        if sid + 1 >= len(first):
            return []
        out = []
        for e in range(first[sid], min(first[sid + 1], len(table))):
            leg = table[e]
            name = f"{what}: {leg.name}" if leg.name else what
            # The edge comes from the table, but the generator's row comments
            # name it too ('PATCH, south node by lane'); do not print it twice.
            edge = "" if leg.edge in name else f", {leg.edge}"
            out += [f"DMA  {arrow} ({name}{edge}, {leg.nbytes} B)"] * leg.pieces
        return out

    def _pushes(self, cid, sid):
        """-> [(nbytes, tensor), ...] the cluster's own hand-off, in issue order."""
        lines = self.push_src.get(sid)
        if not lines:
            return []
        env = dict(self.env, lane=self.cid_lane[cid], cid=cid, stage=sid)
        return self._exec(lines, env, [],
                          f"stage {sid} hand-off (cluster {cid})")

    def _handoff_regions(self, cid, sid):
        """The bracketed DMAs push_to_next() issues for ONE cluster.

        One region per call this lane's push_*() helper makes, in the order it
        makes them, sized by what that one call moves: every lane drives its own
        link, so a 4-lane hand-off is four clusters pushing in parallel, not one
        transfer of their sum.

        Both schedules bracket their pushes -- cwt issues them ahead of the
        compute rather than after it, through the same helper -- so the sequence
        is the same either way.
        """
        kind = self._handoff_kind(sid)
        out = []
        for nbytes, tensor in self._pushes(cid, sid):
            what = f"{tensor}, " if tensor else ""
            size = "?" if nbytes is None else nbytes
            out.append(f"DMA  {kind} -> stage {sid + 1} ({what}{size} B)")
        return out

    def _handoff_kind(self, sid):
        """'push' / 'scatter' / 'gather' for the boundary after stage sid."""
        if sid + 1 >= len(self.lanes):
            return "push"
        lanes, lanes_nx = self.lanes[sid], self.lanes[sid + 1]
        return ("push" if lanes == lanes_nx
                else "scatter" if lanes_nx > lanes else "gather")

    @staticmethod
    def _disambiguate(labels):
        """Two identical regions in one stage are two rows, not one.

        Regions are keyed by label, so a stage running the same kernel on the
        same shape twice would otherwise fold both executions into one row.  The
        suffix goes after the trailing ')', clear of the size and shape patterns
        work_model() reads back out.
        """
        seen = defaultdict(int)
        out = []
        for label in labels:
            seen[label] += 1
            out.append(label if seen[label] == 1 else f"{label} #{seen[label]}")
        return out

    def _cluster_regions(self, cid, sid):
        """The dm and worker region sequences of ONE cluster, in execution order.

        Mirrors main()'s step: the inbound off-chip read (dm only, ahead of the
        intra-cluster sync the workers wait on), then the compute every core
        runs, then the transfer phase -- the outbound legs, then the push -- all
        of it dm only.  Only the push differs between lanes of a stage.

        Both schedules emit this same sequence: cwt only moves item i's drain
        into step i + 1, where the next item's compute overlaps it, so the
        difference between them is WHEN a region runs, not WHETHER.
        """
        dm, worker = [], []
        dm += self._lpddr_regions(sid, self.lpddr_in, self.in_first,
                                  "LPDDR -> L1", "stage input")
        for label, dm_only in self.bodies.get(sid, []):
            dm.append(label)
            if not dm_only:
                worker.append(label)
        dm += self._lpddr_regions(sid, self.lpddr_out, self.out_first,
                                  "L1 -> LPDDR", "stage output")
        dm += self._handoff_regions(cid, sid)
        return self._disambiguate(dm), self._disambiguate(worker)

    @property
    def n_working(self):
        """Clusters the pipeline actually uses -- N_CLUSTERS counts idle cids too."""
        return len(self.tables)

    def stage_of(self, cid):
        """-> the stage on this cid, or None if the pipeline leaves it idle.

        Stage 0 is the generator's spelling of idle, not a stage: a cid carrying
        it runs nothing and belongs in no column of the report.
        """
        if cid >= len(self.cid_stage):
            return None
        return self.cid_stage[cid] or None

    def table_for(self, cid):
        """-> {'dm': [...], 'worker': [...]} for a cluster, or None if unmapped."""
        return self.tables.get(cid)

    def title_of(self, cid):
        stage = self.stages.get(self.stage_of(cid))
        return stage["name"] if stage else "?"


# --------------------------------------------------------------------------
# The celere app model
#
# celere_pipeline.py emits a different app shape from cevit_pipeline.py:
#
#   * stages numbered 1..N_STAGES (slot 0 of every table is dead), not from 0;
#   * the hand-off is a table walk over STAGE_XFER, one row per tensor crossing a
#     boundary -- { bytes, rows, src_off } -- and the two stages' lane counts
#     alone set how many bracketed DMAs each row costs;
#   * off-chip legs come from LPDDR_IN_XFER / LPDDR_OUT_XFER, one bracketed DMA
#     per node an entry spans; either range may be empty for a stage;
#   * a tiled last stage (STAGE_TILES > 1) drains itself and main() issues no
#     push for it.
#
# All of these DMAs bracket themselves, so the DM core's region indices run
# ahead of the worker cores' -- hence the dm/worker split in the region tables.
#
# This tracks the CURRENT generator only: a log from an earlier one reports raw.
# --------------------------------------------------------------------------

# What the two hand-off tables give a DMA row: how many bytes it moves and what
# to call it.  An Xfer's `rows` and `src_off` reach no label (rows becomes the 2D
# descriptor's strides, src_off addresses the producer's own buffer) but both are
# carried, since a row lacking either is not a table this parser can read.
#
# The five trailing fields are the LAYOUT-ROW form the generator grew later, and
# they change the region COUNT, not just a size: nseg > 0 makes the leg walk
# `cols` box columns per consumer and issue SEG[seg0 .. seg0+nseg) inside each,
# which is cols * nseg bracketed DMAs where the plain copy issues one.  They
# default to the plain copy, so an older 3-field row still reads.
Xfer = namedtuple("Xfer", "nbytes rows src_off cols src_pitch dst_pitch "
                          "seg0 nseg name",
                  defaults=(0, 0, 0, 0, 0, None))
# One segment of a layout row's token-order map, as SEG[] lists it: the bytes the
# single 2D descriptor moves (chunk * reps) and what that piece is called.  The
# offsets and strides say WHERE it lands, which no region row carries.
Seg = namedtuple("Seg", "nbytes name")
# One off-chip leg: the bytes ONE descriptor moves, the edge it crosses, and how
# many nodes the entry walks -- lpddr_pull()/lpddr_push() issue one bracketed DMA
# of `nbytes` per node, so `nodes` is a region count, not a size.
Leg = namedtuple("Leg", "nbytes edge nodes name")

# celere's extra kernel, on top of the ones both generators share.  It is
# unscored: work_model() returns None and the row prints '-' rather than invent
# a peak.
CELERE_KERNEL_SPECS = KERNEL_SPECS + [
    # mempool_radix4_cfft_f16p(...), deliberately unsized: the kernel's header is
    # not in this tree, and the region is unscored either way.
    (re.compile(r"mempool_radix4_cfft_f16p\("), lambda n, g, t: f"FFT  {n}"),
    # pool_mean_f16p(src, dst, spans, Nh, Nw, row_stride, cid, nt).  Unsized for
    # the same reason as the FFT: the kernel is not in this tree, so how it deals
    # a span out over the cores is not known and a peak would be invented.  The
    # label carries the span shape only, to tell two pool rows apart.
    (re.compile(rf"pool_mean_f16p\(\s*{B},\s*{B},\s*"
                r"(\d+)u?,\s*(\d+)u?,\s*(\d+)u?"),
     lambda n, g, t: f"POOL {n} {g[0]}x{g[1]}x{g[2]}"),
]

# permute_f16p(src, dst, elems, levels, n_levels, cid, nt) -- the generator's one
# permutation kernel.  The call carries the BLOCK width; how many blocks there are
# is the product of the level counts, and those live in the table named by the
# fourth argument, so the label is finished from the body's own tables (see
# CelereApp._perm_tables).
CELERE_PERMUTE_CALL_RE = re.compile(
    rf"permute_f16p\(\s*{B},\s*{B},\s*(\d+)u?,\s*(\w+)\s*,")
# One `static const perm_level_t NAME[] = { { count, src_stride, dst_stride },
# ... };`.  Only the count is read: the strides place a block, the counts say how
# many there are.
CELERE_PERM_TABLE_RE = re.compile(
    r"\bperm_level_t\s+(\w+)\s*\[\s*\]\s*=\s*\{(.*?)\}\s*;", re.S)
CELERE_PERM_COUNT_RE = re.compile(r"\{\s*(\d+)")


def _celere_short(name):
    """'P1[fftshift + box grouping]' -> 'P1'.

    _comment_name() peels a trailing '[tag]' of a single word, which is how the
    compute kernels spell themselves ('FFT on dim i_2 [fft]').  The permutations
    put a whole phrase in the bracket, which survives that and comes off here to
    keep the label readable.
    """
    return name.split("[", 1)[0].strip() or name


class CelereApp:
    """The region layout of one generated celere app, read from its .c."""

    overlapped = False          # every hand-off is bracketed; see the note above
    flavour = "celere"          # names the default plot directory and file stem
    plot_tag = "CELERE"

    def __init__(self, path):
        src = Path(path).read_text()
        bare = COMMENT_RE.sub(" ", src)

        d = {m.group(1): int(m.group(2)) for m in DEFINE_RE.finditer(bare)}
        self.n_stages = d["N_STAGES"]
        self.n_iter = d.get("N_ITER", 1)

        self.cid_stage = AppModel._array(bare, "CID_STAGE")
        self.cid_lane = AppModel._array(bare, "CID_LANE")
        self.lanes = AppModel._array(bare, "STAGE_LANES")
        self.tiles = AppModel._array(bare, "STAGE_TILES")
        self.xfer_first = AppModel._array(bare, "STAGE_XFER_FIRST")
        self.in_first = AppModel._array(bare, "LPDDR_IN_FIRST")
        self.out_first = AppModel._array(bare, "LPDDR_OUT_FIRST")
        # The struct tables are read out of the SOURCE, not `bare`: each row's
        # trailing comment names the tensor it carries, and that name is what
        # makes a hand-off row readable in the report.
        self.segs = self._seg_table(src)
        self.xfer = self._xfer_table(src, self.segs)
        self.lpddr_in = self._lpddr_table(src, "LPDDR_IN_XFER")
        self.lpddr_out = self._lpddr_table(src, "LPDDR_OUT_XFER")

        self.titles = {int(m.group(1)): m.group(2)
                       for m in STAGE_TITLE_RE.finditer(src)}
        # Sizes reach the DMA helpers as #defines (TILE_OUT_BYTES), and
        # dma_bytes() only reads literals, so fold them in before scanning.
        expanded = expand_defines(src, d)
        self.bodies = {int(m.group(1)): self._body(m.group(2))
                       for m in COMPUTE_FN_RE.finditer(expanded)}

        self.stages = {}
        for sid in range(1, self.n_stages + 1):
            dm, worker = self._stage_regions(sid)
            self.stages[sid] = {"name": self.titles.get(sid, "?"),
                                "dm": dm, "worker": worker}

    # -- source tables ----------------------------------------------------

    @staticmethod
    def _seg_table(src):
        """-> [Seg, ...], one entry per SEG[] row; [] when the app declares none.

        A row is { chunk, src_off, dst_off, src_stride, dst_stride, reps }: one
        segment of a layout row's token-order map, issued as a single bracketed
        2D descriptor moving chunk * reps bytes.  Only that product reaches a
        label, but a row of another shape would mis-count the descriptors a
        layout leg costs, so it is an error rather than a silent zero.
        """
        rows = table_rows(src, "SEG")
        if rows is None:
            return []
        out = []
        for fields, comment in rows:
            vals = [_uint(f) for f in fields]
            if len(vals) != 6 or any(v is None for v in vals):
                raise ValueError(
                    f"SEG[] row {{{', '.join(fields)}}} is not the "
                    f"{{ chunk, src_off, dst_off, src_stride, dst_stride, "
                    f"reps }} this parser reads")
            out.append(Seg(vals[0] * vals[5], comment))
        return out

    @staticmethod
    def _xfer_table(src, segs):
        """-> [Xfer, ...], one entry per STAGE_XFER[] row.

        A row opens with { bytes, rows, src_off }: the WHOLE tensor across both
        stages' lanes, the destination row count, and where the producer holds
        its own share.  The generator has grown the row once, and both widths
        are read -- the older columns keep their meaning:

          3   the original plain copy: one bracketed DMA per consumer
          8   + cols, src_pitch, dst_pitch, seg0, nseg.  nseg > 0 IS A LAYOUT
              ROW -- the leg walks `cols` box columns per consumer and issues
              SEG[seg0 .. seg0+nseg) inside each, so it costs cols * nseg
              descriptors, not one.  nseg == 0 is the plain copy above

        Only byte counts reach a label, but every field is required, so a table
        of another shape is an error, not a silent zero.
        """
        rows = table_rows(src, "STAGE_XFER")
        if rows is None:
            raise ValueError("no STAGE_XFER[] in app source")
        out = []
        for fields, comment in rows:
            vals = [_uint(f) for f in fields]
            if len(vals) not in (3, 8) or any(v is None for v in vals):
                raise ValueError(
                    f"STAGE_XFER[] row {{{', '.join(fields)}}} is neither the "
                    f"3-field {{ bytes, rows, src_off }} nor the 8-field "
                    f"layout-row xfer_t shape this parser reads")
            x = Xfer(*vals, name=comment)
            if x.nseg and x.seg0 + x.nseg > len(segs):
                raise ValueError(
                    f"STAGE_XFER[] row '{comment}' claims SEG"
                    f"[{x.seg0}..{x.seg0 + x.nseg}), but the app declares "
                    f"{len(segs)} segment(s)")
            out.append(x)
        if not out:
            raise ValueError("no rows in STAGE_XFER[]")
        return out

    @staticmethod
    def _lpddr_table(src, name):
        """-> [Leg, ...], one entry per lpddr_xfer_t row.

        A row opens with { lpddr_off, l1_off, lane_step, chunk, lpddr_stride,
        l1_stride, reps }: chunk * reps bytes per descriptor.  The generator has
        grown the row twice, and all three widths are read -- the older columns
        keep their meaning and the new ones only default:

          7   the original row, before edges were a choice: south, one node
          8   + edge, which direction does not imply (a stage drains north and
              still reads its base off a south node)
          10  + nodes, lanes_per_node.  A lane spanning several nodes gets one
              bracketed DMA each, so `nodes` is a region count; lanes_per_node
              only picks the offset inside a shared node

        Anything else is an error rather than a silently mis-sized leg.
        """
        rows = table_rows(src, name)
        if rows is None:
            return []
        out = []
        for fields, comment in rows:
            vals = [_uint(f) for f in fields[:7]] + [_uint(f)
                                                     for f in fields[8:]]
            edge_field = fields[7].upper() if len(fields) > 7 else "SOUTH"
            if (len(fields) not in (7, 8, 10) or any(v is None for v in vals)
                    or (len(fields) > 7 and "LPDDR_EDGE" not in edge_field)):
                raise ValueError(
                    f"{name}[] row {{{', '.join(fields)}}} is not one of the "
                    f"7-, 8- or 10-field lpddr_xfer_t shapes this parser reads")
            edge = "north" if "NORTH" in edge_field else "south"
            nodes = vals[7] if len(fields) == 10 else 1
            out.append(Leg(vals[3] * vals[6], edge, nodes, comment))
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
    def _perm_tables(text):
        """-> {table name: how many blocks its levels enumerate}.

        A permutation is a nest of levels, each repeating the one below it
        `count` times, so the leaves -- the blocks the kernel actually copies --
        are the PRODUCT of the counts.  The tables are function-local and named
        per call site, so this is read per stage body, not once per app.
        """
        out = {}
        for m in CELERE_PERM_TABLE_RE.finditer(text):
            counts = [int(c) for c in CELERE_PERM_COUNT_RE.findall(m.group(2))]
            if counts:
                out[m.group(1)] = math.prod(counts)
        return out

    @staticmethod
    def _permute_label(name, g, tables):
        """-> 'RSHP PAT : RA unpack + patchify 8768x48' for one permute_f16p().

        Blocks x elements-per-block is the shape layout_work() costs, and it is
        the whole tensor: the kernel walks every block the levels enumerate.  A
        table this body did not declare leaves the label unsized rather than
        wrong, which costs the row its utilization and nothing else.
        """
        blocks = tables.get(g[1])
        short = _celere_short(name)
        return f"RSHP {short} {blocks}x{int(g[0])}" if blocks else f"RSHP {short}"

    @staticmethod
    def _body(text):
        """-> [(label, dm_only), ...] for one compute_stage_N() of a celere app.

        The permutation spec is built per body: its label needs the level tables,
        which are function-local and named per call site.
        """
        tables = CelereApp._perm_tables(text)
        specs = CELERE_KERNEL_SPECS + [
            (CELERE_PERMUTE_CALL_RE,
             lambda n, g, t: CelereApp._permute_label(n, g, tables)),
        ]
        lines = CelereApp._fuse_bare_guards(AppModel._logical_lines(text))
        regions, _ = AppModel._scan(lines, 0, False, {"comment": None}, specs)
        return regions

    def _lpddr_regions(self, sid, table, first, arrow, what):
        """The bracketed DMAs lpddr_pull() / lpddr_push() issue for stage sid.

        One region per node per table entry in the stage's range -- the helpers
        bracket each descriptor separately -- and none at all for a stage that
        touches no off-chip node.  The repeats come out identical and are
        numbered by _disambiguate().
        """
        if sid + 1 >= len(first):
            return []
        out = []
        for e in range(first[sid], min(first[sid + 1], len(table))):
            leg = table[e]
            name = f"{what}: {leg.name}" if leg.name else what
            out += [f"DMA  {arrow} ({name}, {leg.edge}, {leg.nbytes} B)"] * leg.nodes
        return out

    def _handoff_regions(self, sid):
        """The bracketed DMAs push_to_next() issues for stage sid.

        One region per dma_2d/dma_2d_strided call it makes, in the order it makes
        them, sized by what ONE cluster moves.  The shape follows from the two
        stages' lane counts alone:

          fan-out  (lanes_nx > lanes)  this lane covers lanes_nx/lanes consumers
                                       and sends each one its whole share
          fan-in   (lanes > lanes_nx)  one transfer of what this lane holds, into
                                       its slot of the consumer's rows
          straight (equal)             one transfer of this lane's share

        A row's `rows` only re-lays the share out in the descriptor's strides, so
        it moves the same bytes either way and does not appear here.  main()
        skips the push for the last stage and for any tiled stage, which drains
        itself.

        A LAYOUT ROW (nseg > 0) overrides all three: the share is not one
        transfer but one PER SEGMENT of every box column, so the count is
        r * cols * nseg and each row is sized by its own segment -- the shape
        of the fan-out only sets r.
        """
        if sid >= self.n_stages or self.tiles[sid] != 1:
            return []
        lanes, lanes_nx = self.lanes[sid], self.lanes[sid + 1]
        out = []
        for e in range(self.xfer_first[sid],
                       min(self.xfer_first[sid + 1], len(self.xfer))):
            x = self.xfer[e]
            own_src = x.nbytes // lanes         # what this cluster holds
            own_dst = x.nbytes // lanes_nx      # what one consumer needs
            what = f" {x.name}" if x.name else ""
            if x.nseg:
                # The nest push_to_next() runs: r consumers, x.cols box columns
                # each, x.nseg descriptors inside a column.  Every one of them is
                # bracketed on its own, so every one of them is a region.
                r = lanes_nx // lanes if lanes_nx > lanes else 1
                for k in range(r):
                    dest = (f"stage {sid + 1} lane {k}" if r > 1
                            else f"stage {sid + 1}")
                    for _ in range(x.cols):
                        for s in range(x.nseg):
                            seg = self.segs[x.seg0 + s]
                            named = f"{what}: {seg.name}" if seg.name else what
                            out.append(f"DMA  layout -> {dest}{named}"
                                       f" ({seg.nbytes} B)")
            elif lanes_nx > lanes:
                out += [f"DMA  scatter -> stage {sid + 1} lane {k}"
                        f"{what} ({own_dst} B)"
                        for k in range(lanes_nx // lanes)]
            elif lanes > lanes_nx:
                out.append(f"DMA  gather -> stage {sid + 1}{what} ({own_src} B)")
            else:
                out.append(f"DMA  push -> stage {sid + 1}{what} ({own_src} B)")
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

    @property
    def n_working(self):
        """Clusters the pipeline actually uses, matching AppModel's own count."""
        return sum(1 for s in self.cid_stage if s)


def detect_flavour(path):
    """-> 'celere' / 'cevit' / None, from the tables the app source declares.

    The generators share some names (push_to_next(), STAGE_BASE[],
    LPDDR_IN_FIRST[]), so the markers are the ones only one of them has: celere
    moves its hand-off through a descriptor table and pulls off-chip with
    lpddr_pull(), cevit runs a loop nest and moves both directions through the
    one lpddr_move().  A file carrying neither (or both) returns None, so the
    caller can ask for the flag rather than fail somewhere less obvious.
    """
    src = COMMENT_RE.sub(" ", Path(path).read_text())
    celere = "STAGE_XFER_FIRST" in src or "lpddr_pull" in src
    cevit = "lpddr_move" in src
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
# have a FLOP reading.  uti doubles as FLOP utilization (the op->FLOP factor of 2
# is the same for both units and cancels), and a TE row matches the ratio
# light_redmule.cpp prints, so it can be checked against a `make runv` trace.
# --------------------------------------------------------------------------

# TE: nb_redmule_tiles x (redmule_height x redmule_width) MACs per cycle, from
# configs/arch_tensorpool.py.  The 16 engines run concurrently on 1/16 of the
# GEMM each.
RM_TILES, RM_H, RM_W = 16, 8, 32
TE_MAC_PER_CYC = RM_TILES * RM_H * RM_W         # 4096 MAC/cyc per cluster

# PE: num_core_per_cluster cores, one 2-lane f16 vector FP op per cycle each.
N_CORES = 256
PE_OPS_PER_CYC = N_CORES                        # 256 vecop/cyc per cluster

# LSU: the patch permutation kernels do no arithmetic (scalar halfword loads and
# stores), so they get a unit of their own: one load/store issue slot per core
# per cycle.  An "OP" is one halfword access; this unit has no FLOP figure.
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
# The off-chip node is spelled "hbm" in the arch config whatever DRAMSys model is
# behind it, so both spellings are matched.  Its peak is read per run from the
# DRAMSys 'MAX BW' footer; the constant below is only the fallback for a log
# without one, and --dram-peak overrides both.
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
# The 'RxC' every other kernel label carries: rows x columns for the PE kernels,
# units x elements-per-unit for the layout ones.
SHAPE_RE = re.compile(r"(\d+)x(\d+)")

# What work_model() returns: uti = ops / (peak_ops * span), ideal = ops/peak_ops.
# For a DMA an "op" is one byte.  ideal and ceiling are None with no known peak.
Work = namedtuple("Work", "kind ideal ceiling ops peak_ops")


def set_dram_peak(override, seen):
    """Fix the off-chip peak for this run, from --dram-peak or the log's footer.

    `seen` is every distinct 'MAX BW ... GB/s' the DRAMSys footer printed, one
    line per channel; normally all the same figure.  Channels that disagree are
    reported rather than averaged, and the largest is kept so the utilizations
    read as a floor.
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

    Both softmax and layernorm walk rows two at a time
    (`for (i = core_id*2; i < M; i += numThreads*2)`), so the M/2 row-pairs are
    dealt out over the cores and the busiest core runs ceil((M/2)/numThreads) of
    them -- the best utilization this distribution allows on a stall-free core.
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
    so elements 16c+8 .. 16c+15 are never touched.  Both kernels therefore leave
    half their tensor unprocessed -- a bug in the kernels, not in this model.
    What is counted here is what the cores actually execute.
    """
    iters = [max(0, -(-(length - BANKED_START * c) // BANKED_STRIDE))
             for c in range(N_CORES)]
    ops = sum(iters) * ops_per_iter
    critical = max(iters) * ops_per_iter
    ideal = ops / PE_OPS_PER_CYC
    return Work("PE", ideal, (ideal / critical if critical else None),
                ops, PE_OPS_PER_CYC)


def layout_work(units, unit_elems):
    """Build a Work for a permutation kernel: patchify, unpatchify, transpose.

    One halfword load and one halfword store per element, no arithmetic, so the
    unit is the LSU.  All of them deal a UNIT axis out over the cores and keep a
    unit whole -- a patch row for the reshapes, a matrix row or column for the
    transpose -- whether they take a contiguous chunk of it (the reshapes) or
    stride through it (`for r = core_id; r < rows; r += numThreads`).  Either
    distribution leaves the busiest core ceil(units/N_CORES) of them, and that
    ratio is the imbalance ceiling.

    Not counted: the divisions each kernel pays on entry to place its chunk, and
    the tail's loop overhead where the unit is not a multiple of the unrolling.
    Both are real cycles, so the reported uti is issue-slot occupancy of the
    memory path and reads low on a short unit.
    """
    ops_per_unit = LSU_OPS_PER_ELEM * unit_elems
    total = units * ops_per_unit
    critical = -(-units // N_CORES) * ops_per_unit      # busiest core's units
    ideal = total / LSU_OPS_PER_CYC
    return Work("LSU", ideal, (ideal / critical if critical else None),
                total, LSU_OPS_PER_CYC)


def work_model(label):
    """-> Work(kind, ideal_cycles, imbalance_ceiling, ops, peak_ops) or None."""
    if label.startswith("DMA"):
        # The size is the LAST 'N B' in the label: a celere row carries the
        # tensor's own name ahead of it, and a name is free to hold a number.
        m = DMA_BYTES_RE.findall(label)
        if not m:
            return None
        nbytes = int(m[-1])
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
        rows, cols = (int(x) for x in SHAPE_RE.search(label).groups())
        return pe_work(rows, cols, layernorm_ops)

    if label.startswith("SMAX"):
        rows, cols = (int(x) for x in SHAPE_RE.search(label).groups())
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
        units, elems = (int(x) for x in SHAPE_RE.search(label).groups())
        return layout_work(units, elems)

    if label.startswith("TRSP"):
        # transpose_f16p splits whichever axis keeps every core busy: rows when
        # there are at least as many as cores, columns otherwise.  The unit is a
        # row of `cols` elements one way and a column of `rows` the other.
        rows, cols = (int(x) for x in SHAPE_RE.search(label).groups())
        return (layout_work(rows, cols) if rows >= N_CORES
                else layout_work(cols, rows))

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
        per-core mean -- is what the region costs the cluster.
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
    mid-stage, so its last clusters hold a PREFIX of their region table; with
    partial_ok those are labelled as far as they got instead of dropping to raw.
    Without it the exact-length check stands, so an accidental app/log mismatch
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
# transfer's duration: the engine reported done once the interconnect accepted
# the data, before it reached the far end.  Flagged, because the number otherwise
# looks like a result.
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


@dataclass
class Agg:
    """Every execution of one region label, summed across the clusters.

    `work` is that label's model, kept whole for the fields that do not add up
    (the unit, its peak, the per-call ideal); the counters below are the ones
    that do.
    """
    work:  "Work"
    calls: int = 0
    span:  int = 0
    ideal: float = 0.0
    ops:   int = 0


def report(clusters, total_ns, app):
    kernel_totals = defaultdict(int)
    stage_span = {}
    impossible = []             # DMA rows whose span cannot be the transfer's

    # The label column is sized to its content, 46 being the floor: cevit labels
    # fit in that, celere's carry a whole layer description.
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
            # Everything else goes in by CLASS, so the spellings of one layout
            # row (RSHP where a kernel ran it, RESHAPE where it was a view) land
            # in one line here and in one colour on the plots.  Same classifier
            # as the plots use, so the two agree row for row.
            kernel_totals[region_class(r.label, w.kind if w else None)] += r.span

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
            a = agg.setdefault(r.label, Agg(w))
            a.calls += 1
            a.span += r.span
            a.ideal += w.ideal or 0.0
            a.ops += w.ops
    for label, a in sorted(agg.items(), key=lambda kv: -kv[1].span):
        w = a.work._replace(ops=a.ops)
        print(f"    {label:<{lw - 2}}{w.kind:>9} {a.calls:>5} {a.span:>8} "
              f"{rate_cells(w, a.span)} "
              f"{fmt_pct(None if w.ideal is None else a.ideal / a.span, 7)}"
              f"{OVER_PEAK_MARK if over_link_peak(w, a.span) else ''}")

    # One summary line per unit, in the order the units first appear above.
    peaks = {}
    for a in agg.values():
        peaks.setdefault(a.work.kind, a.work.peak_ops)
    for kind, peak_ops in peaks.items():
        rows = [a for a in agg.values() if a.work.kind == kind]
        span = sum(a.span for a in rows)
        ideal = sum(a.ideal for a in rows)
        ops = sum(a.ops for a in rows)
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
    # A class is one word, but not always a short one (a DMA carries its link),
    # so 10 is only the floor.
    kw = max([10] + [len(k) for k in kernel_totals])
    for k, v in sorted(kernel_totals.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<{kw}} {v:>10} cyc  {100.0 * v / grand:5.1f}%")
    print(f"    {'TOTAL':<{kw}} {grand:>10} cyc")

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
#   transfer -- DMA:NoC, DMA:LPDDR   (stage hand-offs and the off-chip legs)
#   compute  -- everything else: TE, PE and the DMA:L1 layout copies
#
# A lane's height is the SUM of its region spans, matching the CSV.  Each bar is
# subdivided into its regions, one segment per kernel and per DMA: CTT orders
# them bottom-to-top by execution, CWT splits them over two bars.
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
    "TRSP":      "#c5b0d5",     # layout work like RSHP, hence the same family
    "POOL":      "#bcbd22",
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

# How a class is spelled in the legend, where that differs from the label.
LAYER_LEGEND = {
    "DMA NoC": "DMA remote L1",
}

# Classes that are one kernel wearing two names.  A region that is not a kernel
# call -- a mempool_wait() stand-in, a view -- takes its class from the comment's
# [tag], so the same layout row is spelled RESHAPE there and RSHP where a
# permutation kernel ran it.  They are one class: folding them keeps the legend
# to one entry and one colour for the layout work of a stage.
LAYER_ALIASES = {
    "SPLIT&RESHAPE": "RSHP",
    "RESHAPE": "RSHP",
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
        holds the worker cores on flex_intra_cluster_sync() until it lands, so it
        is serialized in front of the compute and sits outside the max.  Only
        stage 0 has one; every other lane reduces to max(compute, transfer).
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


def schedule_name(overlapped):
    """The schedule's own name, as the report and the warnings spell it."""
    return "compute-while-transfer" if overlapped else "compute-then-transfer"


def is_transfer(kind):
    """True = crosses a link, False = compute (an unmodelled region counts as compute)."""
    return kind is not None and kind.startswith(TRANSFER_KINDS)


def region_class(label, kind):
    """The legend class of one region: its link if it is a DMA, else its kernel.

    Read off the label's first word, then through LAYER_ALIASES, which folds the
    spellings that name the same kernel into one class.
    """
    if kind is not None and kind.startswith("DMA:"):
        return "DMA " + kind.split(":", 1)[1]       # DMA NoC / DMA LPDDR / DMA L1
    head = label.split()
    return LAYER_ALIASES.get(head[0], head[0]) if head else "?"


def build_stages(clusters, app, cycles_per_ms):
    """-> [Stage, ...] from the same {cluster: [Region]} the report walks.

    A region with no work model still lands in a column.  Unscored kernels (the
    FFTs, the rope, the permutations, every mempool_wait stand-in) are compute
    whether or not a peak exists to score them against, so their column is right.
    A DMA whose size did not parse is the real miss -- it crosses a link and is
    counted as compute, moving time from one bar to the other -- so that warns.
    """
    stages = {}
    miscounted = set()
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
            if w is None and r.label.startswith("DMA"):
                miscounted.add(r.label)
            kind = None if w is None else w.kind
            # The stage input is tracked apart because CWT cannot overlap it; its
            # sibling `L1 -> LPDDR (stage output)` shares the unit but IS
            # overlapped, so the label is what separates the two.
            xf = is_transfer(kind)
            is_din = xf and "stage input" in r.label.lower()
            if xf:
                xfer += r.span
                din += r.span if is_din else 0
            else:
                comp += r.span
            layers.append(Layer(region_class(r.label, kind),
                                r.span / cycles_per_ms, xf, is_din))
        stages.setdefault(sidx, Stage(idx=sidx)).lanes.append(Lane(
            cid         = cid,
            name        = cluster,
            compute_ms  = comp / cycles_per_ms,
            transfer_ms = xfer / cycles_per_ms,
            din_ms      = din / cycles_per_ms,
            layers      = layers,
        ))
    if miscounted:
        print(f"warning: {len(miscounted)} DMA region(s) carry no size, so they "
              f"count as compute rather than transfer: {sorted(miscounted)}",
              file=sys.stderr)
    for st in stages.values():
        st.lanes.sort(key=lambda l: l.cid)
    return [stages[k] for k in sorted(stages)]


def plot_cluster_split(label, stages, overlap, out_path, ymax):
    """Two stacked panels of per-cluster wall time, on one shared y axis.

    One x position per PIPELINE STAGE; a head-parallel stage draws its clusters
    SIDE BY SIDE inside a shaded slot, so the picture shows directly that the
    stage's wall is the height of one lane, not their sum.

    Serialized split (CTT) -> stacked bars   (wall = compute + transfer).
    Overlapped split (CWT) -> grouped bars   (wall = max(compute, transfer)).

    The top panel is the plain compute-vs-transfer split; the bottom one is the
    same bars, same heights, subdivided into the regions that make them up, one
    segment per kernel and per DMA in execution order.  Only the totals are
    shared, so the segments need not line up with the colour boundary above.

    In the CWT top panel a lane's `stage input` read is drawn hatched as the
    pedestal both its compute bar and its overlapped hand-off bar stand on: it
    enters the wall as din + max(compute, rest).  Only stage 0 has one; below, it
    is just another LPDDR segment.
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
    # Then the ones the palette has not met, each taking the first free colour of
    # its side's pool.  Must run after every named class has claimed its colour:
    # the pools share entries with LAYER_COLORS (RSHP and the first compute
    # fallback are both #9467bd).
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
        only which kernel, the one above carries the compute/transfer split.
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
    # The test is for the ABSENCE of a stage layout: AppModel and CelereApp both
    # provide one, NoApp is the case that cannot be plotted.
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
            print(f"warning: --plot_{tag.lower()} but the app runs a "
                  f"{schedule_name(app.overlapped)} schedule; drawing it as "
                  f"asked", file=sys.stderr)
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
                              "(1-based stages, LPDDR_IN/OUT legs, push_*() "
                              "loop-nest hand-offs)")
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
                         "<flavour>_bench_plots/<TAG>_CTT_split_<log>.png next "
                         "to the log")
    ap.add_argument("--plot_cwt", nargs="?", const="", metavar="PNG",
                    help="same figure for the compute-while-transfer flavour "
                         "(grouped bars, with the unoverlappable stage input drawn "
                         "hatched under both)")
    ap.add_argument("--ymax", type=float, metavar="MS",
                    help="fixed y-axis limit in ms, to put two runs on the same "
                         "scale (default: auto-scale)")
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
        try:
            app = (CelereApp if args.flavour == "celere" else AppModel)(args.app)
        except (ValueError, KeyError) as e:
            # A table this parser does not read is the likely cause (both models
            # track the CURRENT generator only); say so rather than let a table
            # regex surface as a traceback.
            print(f"error: cannot read {args.app} as a {args.flavour} app: {e}\n"
                  f"       (this parser reads the current generator's tables; "
                  f"re-run the log's app through the generator, or use --raw)",
                  file=sys.stderr)
            return 1
        # A pipeline need not fill the cids it spans (a narrow stage leaves holes
        # in the grid), so say both counts when they differ.
        cids = len(app.cid_stage)
        span = "" if app.n_working == cids else f" over {cids} cids"
        print(f"app {args.app} [{args.flavour}]: {app.n_stages} stages, "
              f"{app.n_working} clusters{span}, "
              f"{schedule_name(app.overlapped)} schedule")
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
    report(clusters, total_ns, app)
    if args.csv:
        write_csv(clusters, args.csv, app)
        print(f"\nwrote {args.csv}")
    make_plots(clusters, app, args, incomplete)
    return 0


if __name__ == "__main__":
    sys.exit(main())
