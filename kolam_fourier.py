"""
kolam_fourier.py
================
Turn a photograph into a pulli kolam whose dots carry its Fourier transform.

Pipeline (one function group per phase):

  Phase 1  Transform the image
      load_image -> compute_spectrum -> estimate_lattice -> sample_spectrum
      -> symmetrise -> normalise -> quantise              (make_levels wraps these)

  Phase 2  Set the gates
      build_grid -> close_rim -> shell_sweep -> rescue_sealed   (assign_gates wraps these)

  Phase 3  Draw
      gate_matrix -> next_step -> trace_loop -> trace_all -> plot_loops / render_fill

Coordinates
-----------
* Spectrum:   (u, v) integer sample indices, u across, v up.  DC is (0, 0).
* Paper frame (Sivakumar & Sivakumar 2026): dots at even (i, j) with (i + j) % 4 == 0,
  gates at odd (i, j).  A dot's four gates are at (i +/- 1, j +/- 1).
* Drawing frame: the paper frame rotated 45 degrees, x = (i + j) / 2, y = (i - j) / 2.
  Dots then sit on a square grid with a gate between every pair of neighbours.
  Dot (i, j) carries spectrum sample (u, v) = (x / 2, y / 2).

Only numpy, scipy, matplotlib and Pillow are needed.
"""
from __future__ import annotations

import itertools
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.ndimage import gaussian_filter, maximum_filter

# ----------------------------------------------------------------------------
# look
# ----------------------------------------------------------------------------
BG = "#14100c"          # background
FG = "#f3ead6"          # lines and text
DIM = "#4a4238"         # dots not yet reached
YEL, TEAL, RED, ORANGE = "#f5c518", "#35e0c0", "#ff4d4d", "#ff8a3d"
LEVEL_COLOURS = {1: "#5a4bd1", 2: "#c2185b", 3: "#f57c17", 4: "#f5c518"}
LOOP_COLOURS = ["#f5c518", "#35e0c0", "#ff7ab8", "#8fd14f", "#ff8a3d", "#7aa2ff",
                "#e8e3d8", "#c77dff", "#ff5d5d", "#4fd1ff", "#ffd28a", "#b8f06a",
                "#ff9ecb", "#9d8cff", "#62f2c8"]
HEADING = {0: (1, -1), 1: (1, 1), 2: (-1, 1), 3: (-1, -1)}   # drawing-frame direction


def dark_style():
    """Matplotlib settings used by every figure."""
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG,
                         "savefig.facecolor": BG, "text.color": FG,
                         "axes.labelcolor": FG, "xtick.color": FG, "ytick.color": FG,
                         "axes.edgecolor": "#6b6257", "font.size": 12})


def _new(w=7, h=7, n=1, m=1, **kw):
    fig, ax = plt.subplots(n, m, figsize=(w, h), **kw)
    for a in np.atleast_1d(ax).ravel():
        a.set_facecolor(BG)
    return fig, ax


def to_draw(p):
    """Paper frame (i, j) -> drawing frame (x, y)."""
    return ((p[0] + p[1]) / 2, (p[0] - p[1]) / 2)


# ============================================================================
# PHASE 1   Transform the image
# ============================================================================
def load_image(path, size=512):
    """Step 1. Greyscale, centred square crop, resized to size x size, values 0..1."""
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    n = min(a.shape)
    oy, ox = (a.shape[0] - n) // 2, (a.shape[1] - n) // 2
    a = a[oy:oy + n, ox:ox + n]
    if size and n != size:
        a = np.asarray(Image.fromarray((a * 255).astype(np.uint8))
                       .resize((size, size), Image.LANCZOS), dtype=np.float32) / 255.0
    return a


def compute_spectrum(img, pad_factor=8):
    """Step 2. Mean-subtract, Hann window, zero-pad, 2-D FFT centred on DC.

    Zero-padding adds no information; it interpolates the spectrum on a grid
    pad_factor times finer so peaks can be located and sampled precisely.
    The true origin (DC) is at index pad // 2.
    """
    n = img.shape[0]
    z = (img - img.mean()) * np.outer(np.hanning(n), np.hanning(n))
    pad = n * pad_factor
    buf = np.zeros((pad, pad), np.float32)
    o = (pad - n) // 2
    buf[o:o + n, o:o + n] = z
    F = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(buf))).astype(np.complex64)
    return dict(F=F, amp=np.abs(F), pad=pad, cc=pad // 2, up=float(pad_factor), native=n, img=img)


def estimate_lattice(sp, min_radius_frac=0.02, max_radius_frac=0.45):
    """Step 3. Spacing of the reciprocal lattice along each axis, in padded pixels.

    Finds local maxima of the smoothed log amplitude and returns the distance to
    the strongest one lying on the horizontal axis (a*) and on the vertical axis (b*).
    contrast = strongest peak over median background; below ~1.5 there is no lattice.
    """
    pad, cc = sp["pad"], sp["cc"]
    r0, r1 = int(pad * min_radius_frac / 2), int(pad * max_radius_frac)
    s = gaussian_filter(np.log1p(sp["amp"][cc - r1:cc + r1, cc - r1:cc + r1]), 2.0)
    m = s.shape[0] // 2
    yy, xx = np.mgrid[:s.shape[0], :s.shape[0]]
    rr = np.hypot(yy - m, xx - m)
    ys, xs = np.nonzero((s == maximum_filter(s, 21)) & (rr > r0) & (rr < r1))
    vals = s[ys, xs]
    dy, dx = ys - m, xs - m

    def axis(off, along):
        sel = np.abs(off) <= 3
        return float(abs(along[sel][np.argmax(vals[sel])])) if sel.any() else None

    return dict(a_star=axis(dy, dx), b_star=axis(dx, dy),
                contrast=float(vals.max() / max(np.median(s[rr < r1]), 1e-9)),
                a_star_native=axis(dy, dx) / sp["up"], b_star_native=axis(dx, dy) / sp["up"])


def diamond_points(ND):
    """The (ND^2 + 1) / 2 sample indices (u, v) with |u| + |v| <= (ND - 1) / 2."""
    R = (ND - 1) // 2
    return [(u, v) for u in range(-R, R + 1) for v in range(-R, R + 1) if abs(u) + abs(v) <= R]


def sample_spectrum(sp, lattice, ND=None, sub=3, pts=None):
    """Step 4. Read log amplitude and phase at the diamond of points.

    Points are spaced a*/sub horizontally and b*/sub vertically, so every sub-th
    point lands exactly on a diffraction order and the rest fall between orders.
    """
    pts = list(pts) if pts is not None else diamond_points(ND)
    a, b, cc = lattice["a_star"], lattice["b_star"], sp["cc"]
    pix = {p: (int(round(cc + p[1] * b / sub)), int(round(cc + p[0] * a / sub))) for p in pts}
    return dict(pts=pts, sub=sub, a=a, b=b, pix=pix,
                logamp={p: float(np.log10(sp["amp"][pix[p]] + 1e-12)) for p in pts},
                phase={p: float(np.angle(sp["F"][pix[p]])) for p in pts},
                on_lattice={p: (p[0] % sub == 0 and p[1] % sub == 0) for p in pts})


def symmetrise(values):
    """Step 5. Average each value with its mirror images at (+-u, +-v)."""
    return {p: float(np.mean([values[q] for q in {(su * p[0], sv * p[1]) for su in (1, -1) for sv in (1, -1)}
                              if q in values])) for p in values}


def normalise(values, exclude_dc=True):
    """Step 6a. Scale to 0..1. DC is left out of the scaling and set to 1."""
    rest = [v for p, v in values.items() if not (exclude_dc and p == (0, 0))]
    lo, hi = min(rest), max(rest)
    out = {p: (v - lo) / (hi - lo) for p, v in values.items()}
    if exclude_dc and (0, 0) in out:
        out[(0, 0)] = 1.0
    return out


def quantise(norm, n_levels=4):
    """Step 6b. Equal-width bins -> level 1..n_levels."""
    return {p: int(np.clip(np.floor(min(g, 0.9999) * n_levels) + 1, 1, n_levels))
            for p, g in norm.items()}


def make_levels(sp, lattice, ND=None, sub=3, pts=None):
    """Steps 4-6 in one call. Returns the sample dict extended with sym, norm, level."""
    S = sample_spectrum(sp, lattice, ND, sub, pts)
    S["sym"] = symmetrise(S["logamp"])
    S["norm"] = normalise(S["sym"])
    S["level"] = quantise(S["norm"])
    return S


# ============================================================================
# PHASE 2   Set the gates
# ============================================================================
def build_grid(ND):
    """Step 7. Dots, gates, which dots share each gate, the rim, and shells."""
    dots = [(i, j) for i in range(-ND + 1, ND, 2) for j in range(-ND + 1, ND, 2) if (i + j) % 4 == 0]
    uv = {d: ((d[0] + d[1]) // 4, (d[0] - d[1]) // 4) for d in dots}
    owner = {}
    for d in dots:
        for g in gates_of(d):
            owner.setdefault(g, []).append(d)
    rim = {g for g, o in owner.items() if len(o) == 1}
    return dict(ND=ND, R=(ND - 1) // 2, dots=dots, uv=uv, owner=owner, rim=rim,
                shell={d: abs(uv[d][0]) + abs(uv[d][1]) for d in dots})


def gates_of(d):
    """The four gates at a dot's corners (paper frame)."""
    i, j = d
    return [(i - 1, j - 1), (i - 1, j + 1), (i + 1, j - 1), (i + 1, j + 1)]


def mirror_group(p):
    """Step 10. A gate (or dot) and its images under both mirrors (paper frame)."""
    i, j = p
    return {(i, j), (j, i), (-j, -i), (-i, -j)}


def dot_levels(grid, G):
    """Drawn level of every dot = number of its open gates."""
    return {d: sum(G.get(g, 0) for g in gates_of(d)) for d in grid["dots"]}


def close_rim(grid):
    """Step 9. Every gate on the outer edge is closed. Returns the starting gate dict."""
    return {g: 0 for g in grid["rim"]}


def shell_sweep(grid, want, G, record=True):
    """Step 11. From DC outwards, each dot opens as many still-free gates as it needs.

    Gates are always set in mirror groups, so the result is symmetric.
    Gates fixed by inner shells are never changed; no repair is attempted.
    Returns the gate dict and, if record, a snapshot after every shell.
    """
    G = dict(G)
    snaps, done = {}, set()

    def set_group(g, val):
        for h in mirror_group(g):
            G[h] = val

    for s in range(grid["R"] + 1):
        for d in sorted(x for x in grid["dots"] if grid["shell"][x] == s):
            if d in done:
                continue
            done |= mirror_group(d)
            gs = gates_of(d)
            have = sum(G[g] for g in gs if g in G)
            groups = {}
            for g in (g for g in gs if g not in G):
                groups.setdefault(min(mirror_group(g)), []).append(g)
            keys, need, best = list(groups), want[d] - have, None
            for r in range(len(keys) + 1):                 # choose which free groups to open
                for sub in itertools.combinations(keys, r):
                    got = sum(len(groups[k]) for k in sub)
                    score = (abs(got - need), got > need)
                    if best is None or score < best[0]:
                        best = (score, sub)
            for k in keys:
                set_group(k, 1 if k in best[1] else 0)
        if record:
            snaps[s] = dict(G)
    return G, snaps


def rescue_sealed(grid, G, want=None, symmetric=True):
    """Step 12. A dot with no open gate would be a sealed circle: open one of its gates.

    If want is given, open the gate shared with the neighbour furthest below its level."""
    G, rescued = dict(G), []
    for d in sorted(grid["dots"]):
        lv = dot_levels(grid, G)
        if lv[d] == 0:
            cand = [g for g in gates_of(d) if g not in grid["rim"]]
            if want is not None:
                other = lambda g: next(o for o in grid["owner"][g] if o != d)
                cand.sort(key=lambda g: lv[other(g)] - want[other(g)])
            g = cand[0]
            for h in (mirror_group(g) if symmetric else {g}):
                G[h] = 1
            rescued.append(d)
    return G, rescued


def assign_gates(grid, levels_uv):
    """Steps 9-12. levels_uv maps (u, v) -> wanted level. Returns everything recorded."""
    want = {d: levels_uv[grid["uv"][d]] for d in grid["dots"]}
    G0 = close_rim(grid)
    G1, snaps = shell_sweep(grid, want, G0)
    G2, rescued = rescue_sealed(grid, G1)
    drawn = dot_levels(grid, G2)
    return dict(want=want, rim=G0, snaps=snaps, G=G2, rescued=rescued, drawn=drawn,
                wrong=[d for d in grid["dots"] if drawn[d] != want[d]])


# ============================================================================
# PHASE 3   Draw
# ============================================================================
def gate_matrix(G, ND):
    """Gate dict -> (ND+1) x (ND+1) array of 0/1, as in the paper."""
    A = np.zeros((ND + 1, ND + 1))
    for (i, j), v in G.items():
        A[(i + ND) // 2, (j + ND) // 2] = v
    return A


def next_step(A, ND, i, j, heading):
    """Step 13. The paper's tracing rule (eqs. 6-11): one move from gate (i, j).

    Open gate: continue straight to the next gate.  Closed gate: turn 90 degrees
    around the nearest dot.  Returns (next i, next j, next heading, point x, point y)
    where the point is where the curve actually passes, in the paper frame.
    """
    ix, jx = i + ND, j + ND
    alpha = heading % 2
    beta = -1 if heading > 1 else 1
    gamma = -1 if (ix + jx) % 4 == 0 else 1
    g = 1 if A[ix // 2, jx // 2] > 0.5 else 0
    gd, ad = 1 - g, 1 - alpha
    n_alpha = g * alpha + gd * ad
    n_beta = (g + gd * gamma) * beta
    di = (ad * gamma * gd + alpha * g) * beta
    dj = (alpha * gamma * gd + ad * g) * beta
    pi = i + gd * (ad * gamma - alpha) * beta * 0.5
    pj = j + gd * (alpha * gamma - ad) * beta * 0.5
    n_head = (0 if n_beta == 1 else 2) if n_alpha == 0 else (1 if n_beta == 1 else 3)
    return int(i + 2 * di), int(j + 2 * dj), n_head, pi, pj


def trace_loop(A, ND, start):
    """Step 14-15. Follow the rule from start = (i, j, heading) until it returns."""
    st, states, pts = start, [], []
    while True:
        states.append(st)
        ni, nj, nh, pi, pj = next_step(A, ND, *st)
        pts.append((pi, pj))
        st = (ni, nj, nh)
        if st == start:
            return dict(start=start, states=states, pts=np.array(pts))


def trace_all(grid, G, first=None, seed=0):
    """Step 16. Trace loops until every open gate has been crossed in both directions.

    Each open gate carries two strands (heading 0/2 and 1/3). A new loop starts
    at an open gate with an uncrossed strand, chosen at random.
    """
    ND = grid["ND"]
    A = gate_matrix(G, ND)
    rng = np.random.default_rng(seed)
    opens = [g for g, v in G.items() if v == 1]
    used, loops, start = set(), [], first
    while True:
        if start is None:
            cand = [(g, c) for g in opens for c in (0, 1) if (g, c) not in used]
            if not cand:
                return loops
            g, c = cand[rng.integers(len(cand))]
            start = (g[0], g[1], int(c) + 2 * int(rng.integers(2)))
        loop = trace_loop(A, ND, start)
        for (i, j, h) in loop["states"]:
            if G.get((i, j), 0) == 1:
                used.add(((i, j), h % 2))
        loops.append(loop)
        start = None


def largest_loop_start(grid, G):
    """A start on the biggest loop, at its open gate nearest the centre (good for teaching)."""
    big = max(trace_all(grid, G), key=lambda l: len(l["pts"]))
    cand = [s for s in big["states"] if G.get((s[0], s[1]), 0) == 1]
    return min(cand, key=lambda s: (abs(s[0] + s[1]) + abs(s[0] - s[1]), s))


def loop_xy(loop):
    """Pen positions of a loop in the drawing frame."""
    p = loop["pts"]
    return np.column_stack([(p[:, 0] + p[:, 1]) / 2, (p[:, 0] - p[:, 1]) / 2])


def chaikin(P, rounds=4, closed=True):
    """Corner-cutting smoothing. Duplicated vertices stay sharp."""
    P = np.asarray(P, float)
    for _ in range(rounds):
        if closed:
            A_, B_ = P, np.roll(P, -1, 0)
            Q = np.empty((2 * len(P), 2))
            Q[0::2], Q[1::2] = .75 * A_ + .25 * B_, .25 * A_ + .75 * B_
        else:
            Q = [P[0]]
            for a, b in zip(P[:-1], P[1:]):
                Q += [.75 * a + .25 * b, .25 * a + .75 * b]
            Q = np.array(Q + [P[-1]])
        P = Q
    return np.r_[P, P[:1]] if closed else P


# ============================================================================
# FIGURES
# ============================================================================
def plot_image(img):
    fig, ax = _new(6, 6)
    ax.imshow(img, cmap="gray"); ax.axis("off")
    return fig


def _log_amp(sp):
    return np.log10(sp["amp"] + 1e-12)


def _extent(sp):
    c, p, u = sp["cc"], sp["pad"], sp["up"]
    return [-c / u, (p - c) / u, -c / u, (p - c) / u]


def plot_spectrum(sp, zoom=160):
    L = _log_amp(sp)
    fig, ax = _new(6.5, 6.5)
    ax.imshow(L, cmap="inferno", origin="lower", extent=_extent(sp),
              vmin=np.percentile(L, 40), vmax=L.max())
    ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom)
    ax.set_xlabel("u  (cycles across the image)"); ax.set_ylabel("v  (cycles up the image)")
    return fig


def plot_lattice(sp, lat, orders=4, zoom=70):
    L = _log_amp(sp)
    a, b = lat["a_star_native"], lat["b_star_native"]
    fig, ax = _new(6.5, 6.5)
    ax.imshow(L, cmap="inferno", origin="lower", extent=_extent(sp),
              vmin=np.percentile(L, 40), vmax=L.max())
    for h in range(-orders, orders + 1):
        for k in range(-orders, orders + 1):
            ax.plot(h * a, k * b, "o", ms=13, mfc="none", mec=TEAL, mew=1.3)
    for tip, lab, pos in [((a, 0), f"a* = {a:.1f}", (a / 2, -7)), ((0, b), f"b* = {b:.1f}", (3, b / 2))]:
        ax.annotate("", xy=tip, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=YEL, lw=2.5))
        ax.text(*pos, lab, color=YEL, fontsize=13, fontweight="bold")
    ax.set_xlim(-zoom, zoom); ax.set_ylim(-zoom, zoom); ax.set_xlabel("u"); ax.set_ylabel("v")
    return fig


def plot_sampling(sp, S):
    L = _log_amp(sp)
    fig, ax = _new(6.5, 7)
    ax.imshow(L, cmap="inferno", origin="lower", extent=_extent(sp),
              vmin=np.percentile(L, 40), vmax=L.max())
    k = sp["up"] * S["sub"]
    xy = lambda ps: np.array([(p[0] * S["a"] / k, p[1] * S["b"] / k) for p in ps])
    off = xy([p for p in S["pts"] if not S["on_lattice"][p]])
    on = xy([p for p in S["pts"] if S["on_lattice"][p]])
    ax.scatter(off[:, 0], off[:, 1], s=9, c=TEAL, alpha=.8, linewidths=0)
    ax.scatter(on[:, 0], on[:, 1], s=80, facecolors="none", edgecolors=TEAL, linewidths=1.6)
    m = np.abs(np.r_[off, on]).max() + 5
    ax.set_xlim(-m, m); ax.set_ylim(-m, m); ax.set_xlabel("u"); ax.set_ylabel("v")
    return fig


def plot_symmetrise(S):
    pts = S["pts"]; X = [p[0] for p in pts]; Y = [p[1] for p in pts]
    raw = np.array([S["logamp"][p] for p in pts]); sym = np.array([S["sym"][p] for p in pts])
    lo, hi = np.percentile(np.r_[raw, sym], 2), np.r_[raw, sym].max()
    fig, axs = _new(15, 5.2, 1, 3)
    for ax, v, t in [(axs[0], raw, "measured"), (axs[1], sym, "symmetrised")]:
        ax.scatter(X, Y, c=v, cmap="inferno", s=42, marker="s", vmin=lo, vmax=hi)
        ax.set_title(t)
    d = sym - raw
    axs[2].scatter(X, Y, c=d, cmap="coolwarm", s=42, marker="s", vmin=-abs(d).max(), vmax=abs(d).max())
    axs[2].set_title("what changed")
    for ax in axs:
        ax.set_aspect("equal"); ax.axis("off")
    return fig


def plot_quantise(S):
    pts = S["pts"]
    g = np.array([S["norm"][p] for p in pts if p != (0, 0)])
    fig, axs = _new(13, 5.6, 1, 2, gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axs[0]
    for k in range(4):
        sel = (g >= k / 4) & ((g < (k + 1) / 4) | (k == 3))
        ax.hist(g[sel], bins=np.linspace(k / 4, (k + 1) / 4, 11), color=LEVEL_COLOURS[k + 1], edgecolor=BG)
    top = ax.get_ylim()[1]
    for k in range(4):
        if k:
            ax.axvline(k / 4, color=FG, lw=1, ls="--")
        ax.text(k / 4 + .125, top * .93, f"level {k + 1}", ha="center", fontweight="bold")
    ax.set_xlim(0, 1); ax.set_xlabel("normalised log amplitude"); ax.set_ylabel("number of points")
    ax = axs[1]
    lev = np.array([S["level"][p] for p in pts])
    ax.scatter([p[0] for p in pts], [p[1] for p in pts], c=lev, s=40, marker="s", vmin=.5, vmax=4.5,
               cmap=ListedColormap([LEVEL_COLOURS[k] for k in (1, 2, 3, 4)]))
    ax.set_aspect("equal"); ax.axis("off")
    return fig


def plot_grid(grid):
    """Step 7 figure: paper frame and drawing frame side by side."""
    fig, axs = _new(14, 7, 1, 2)
    d0 = (0, 0)
    for ax, f, title in [(axs[0], lambda p: p, "paper layout: dots on even sites, alternate diagonals;\ngates on odd sites"),
                         (axs[1], to_draw, "rotated 45°, as drawn: dots on a square grid,\none gate between every pair of neighbours")]:
        for g in grid["owner"]:
            ax.plot(*f(g), "o", ms=6, mfc="none", mec=FG, mew=1)
        for d in grid["dots"]:
            ax.plot(*f(d), "o", ms=10, color=YEL)
        for g in gates_of(d0):
            ax.plot(*zip(f(d0), f(g)), c=TEAL, lw=2); ax.plot(*f(g), "o", ms=8, color=TEAL)
        ax.set_aspect("equal"); ax.axis("off"); ax.set_title(title, fontsize=12)
    return fig


def plot_primitives():
    """Step 8 figure: the five legal shapes (and the forbidden circle), traced by the real rule."""
    nd, C = 7, [(-1, -1), (1, -1), (1, 1), (-1, 1)]
    cases = [("circle", [], 0), ("drop", [1], 1), ("door", [1, 2], 2), ("eye", [1, 3], 2),
             ("fan", [1, 2, 3], 3), ("diamond", [0, 1, 2, 3], 4)]
    fig, axs = _new(21, 4.4, 1, 6)
    for ax, (name, opened, lv) in zip(axs, cases):
        G = {g: 0 for g in _all_gates(nd)}
        for k in opened:
            G[C[k]] = 1
        A, seen = gate_matrix(G, nd), set()
        for st in [(c[0], c[1], h) for c in C for h in range(4)]:
            lp = trace_loop(A, nd, st)
            key = frozenset(map(tuple, np.round(lp["pts"], 3)))
            if key in seen:
                continue
            seen.add(key)
            xy = loop_xy(lp)
            if np.hypot(*xy.T).min() < 1.2:
                S = chaikin(xy)
                if lv:
                    ax.fill(S[:, 0], S[:, 1], color=LEVEL_COLOURS[lv], alpha=.9)
                ax.plot(S[:, 0], S[:, 1], color=FG, lw=3.2)
        _draw_gates(ax, {c: G[c] for c in C}, ms=15)
        ax.plot(0, 0, "o", ms=9, color="white" if lv == 0 else BG, zorder=6)
        ax.set_xlim(-1.7, 1.7); ax.set_ylim(-1.7, 1.7); ax.set_aspect("equal"); ax.axis("off")
        ax.set_title(f"{name}\n{len(opened)} open" + ("  (forbidden)" if lv == 0 else f"  →  level {lv}"),
                     color=RED if lv == 0 else FG, fontsize=14)
    return fig


def _all_gates(ND):
    return [(i, j) for i in range(-ND, ND + 1, 2) for j in range(-ND, ND + 1, 2)]


def _draw_gates(ax, G, ms=10, frame=to_draw, highlight=None):
    """Open gate: hollow circle. Closed gate: grey disc with a cross. Unset: faint ring."""
    highlight = highlight or {}
    for g, v in G.items():
        X, Y = frame(g); c = highlight.get(g)
        if v is None:
            ax.plot(X, Y, "o", ms=ms * .7, mfc="none", mec=c or DIM, mew=2.2 if c else 1, zorder=3)
        elif v:
            ax.plot(X, Y, "o", ms=ms, mfc=BG, mec=c or FG, mew=2.4 if c else 1.5, zorder=3)
        else:
            ax.plot(X, Y, "o", ms=ms, mfc="#8d8375", mec=c or "#8d8375", mew=2.4 if c else 1, zorder=3)
            k = .014 * ms
            ax.plot([X - k, X + k], [Y - k, Y + k], c=BG, lw=1.6, zorder=4)
            ax.plot([X - k, X + k], [Y + k, Y - k], c=BG, lw=1.6, zorder=4)


def plot_board(grid, G, levels=None, ax=None, dim=(), xmark=(), ring=(), highlight=None,
               show_numbers=True, sc=1.0, title=None):
    """Dots coloured by level (numbers on them) and gates in their current state."""
    if ax is None:
        fig, ax = _new(8, 8)
    else:
        fig = ax.figure
    Gshow = {g: G.get(g) for g in grid["owner"]}
    _draw_gates(ax, Gshow, ms=10 * sc, highlight=highlight)
    for d in grid["dots"]:
        X, Y = to_draw(d); lv = levels[d] if levels else None
        faded = d in dim
        ax.plot(X, Y, "o", ms=15 * sc, color=DIM if faded or lv is None else LEVEL_COLOURS[max(1, lv)], zorder=5)
        if show_numbers and lv is not None:
            ax.text(X, Y, str(lv), ha="center", va="center", fontsize=9 * sc, fontweight="bold", zorder=6,
                    color="#9a9086" if faded else (BG if lv == 4 else "white"))
        if d in xmark:
            ax.plot(X, Y, marker="x", ms=17 * sc, mew=3 * sc, color=RED, zorder=8)
        if d in ring:
            ax.plot(X, Y, "o", ms=24 * sc, mfc="none", mec=TEAL, mew=2.4, zorder=7)
    E = grid["ND"] + .8
    ax.set_xlim(-E, E); ax.set_ylim(-E, E); ax.set_aspect("equal"); ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14)
    return fig


def plot_mirror_groups(grid, G, seeds=((1, 3), (3, 5), (5, 1))):
    hi, cols = {}, [TEAL, YEL, "#ff7ab8"]
    for c, s in zip(cols, seeds):
        for g in mirror_group(s):
            hi[g] = c
    fig = plot_board(grid, G, dim=set(grid["dots"]), show_numbers=False, highlight=hi)
    ax = fig.axes[0]; E = grid["ND"] + .8
    ax.plot([-E, E], [0, 0], "--", c="#8d8375", lw=1.2, zorder=0)
    ax.plot([0, 0], [-E, E], "--", c="#8d8375", lw=1.2, zorder=0)
    for c, s in zip(cols, seeds):
        for g in mirror_group(s):
            ax.plot(*to_draw(g), "o", ms=14, color=c, zorder=9)
    return fig


def plot_sweep(grid, want, snaps, stages=None):
    stages = stages or sorted({0, 1, grid["R"] // 2, grid["R"]})
    fig, axs = _new(15, 15, 2, 2, gridspec_kw={"wspace": 0, "hspace": .08})
    for ax, s in zip(np.ravel(axs), stages):
        G = snaps[s]
        done = {d for d in grid["dots"] if grid["shell"][d] <= s}
        lv = dot_levels(grid, G)
        short = {d for d in done if lv[d] != want[d]}
        plot_board(grid, G, levels=want, ax=ax, dim=set(grid["dots"]) - done, xmark=short, sc=.95,
                   title=f"after shell {s}" + ("  (DC)" if s == 0 else "") + f"  ·  {len(done)} settled, {len(short)} short")
        if s:
            r = 2 * s
            ax.plot([r, 0, -r, 0, r], [0, r, 0, -r, 0], "--", c=TEAL, lw=2, zorder=1)
    return fig


def _trace_board(ax, grid, G, gs=8, ds=7):
    for g in grid["owner"]:
        X, Y = to_draw(g)
        if G.get(g, 0):
            ax.plot(X, Y, "o", ms=gs, mfc=BG, mec="#9c9284", mew=1.1, zorder=3)
        else:
            ax.plot(X, Y, "o", ms=gs, color="#4d453c", zorder=3)
            k = .013 * gs
            ax.plot([X - k, X + k], [Y - k, Y + k], c=BG, lw=1.4, zorder=4)
            ax.plot([X - k, X + k], [Y + k, Y - k], c=BG, lw=1.4, zorder=4)
    for d in grid["dots"]:
        ax.plot(*to_draw(d), "o", ms=ds, color="#e8e0d2", zorder=5)
    E = grid["ND"] + .8
    ax.set_xlim(-E, E); ax.set_ylim(-E, E); ax.set_aspect("equal"); ax.axis("off")


def _start_mark(ax, start, colour):
    X, Y = to_draw((start[0], start[1])); dx, dy = HEADING[start[2]]
    ax.plot(X, Y, marker="*", ms=26, color=colour, mec=BG, mew=1.2, zorder=12)
    ax.annotate("", xy=(X + 1.1 * dx, Y + 1.1 * dy), xytext=(X, Y), zorder=11,
                arrowprops=dict(arrowstyle="-|>,head_width=0.5,head_length=0.8", color=colour, lw=3.2))


def plot_rule(grid, G, loop):
    """Step 13 figure: one loop with its open (teal) and closed (orange) gates ringed."""
    fig, ax = _new(8, 8)
    _trace_board(ax, grid, G, gs=22, ds=15)
    xy = loop_xy(loop); S = chaikin(xy)
    ax.plot(S[:, 0], S[:, 1], color=YEL, lw=5, zorder=6)
    ax.plot(xy[:, 0], xy[:, 1], "o", ms=8, color=RED, zorder=9)
    for (i, j, _) in loop["states"]:
        ax.plot(*to_draw((i, j)), "o", ms=34, mfc="none", mec=TEAL if G.get((i, j), 0) else ORANGE, mew=3, zorder=8)
    _start_mark(ax, loop["start"], YEL)
    lo, hi = xy.min(0) - 1.6, xy.max(0) + 1.6; c = (lo + hi) / 2; h = max(hi - lo) / 2
    ax.set_xlim(c[0] - h, c[0] + h); ax.set_ylim(c[1] - h, c[1] + h)
    return fig


def plot_trace(grid, G, loops_done=(), current=None, steps=None, colour=YEL):
    """Steps 14-16 figures: earlier loops faded, the current one drawn (partly or whole)."""
    fig, ax = _new(8, 8)
    _trace_board(ax, grid, G)
    for lp in loops_done:
        S = chaikin(loop_xy(lp)); ax.plot(S[:, 0], S[:, 1], color=YEL, lw=3, alpha=.35, zorder=6)
    if current is not None:
        xy = loop_xy(current)
        S = chaikin(xy[:steps], closed=False) if steps else chaikin(xy)
        ax.plot(S[:, 0], S[:, 1], color=colour, lw=3.5, zorder=7)
        _start_mark(ax, current["start"], colour)
    return fig


def plot_loops(grid, loops, show_grid=False, lw=2.4):
    """Step 17 figure: every loop in its own colour."""
    fig, ax = _new(9, 9)
    if show_grid:
        _trace_board(ax, grid, {}, gs=0, ds=6)
    for n, lp in enumerate(sorted(loops, key=lambda l: -len(l["pts"]))):
        S = chaikin(loop_xy(lp)); ax.plot(S[:, 0], S[:, 1], color=LOOP_COLOURS[n % len(LOOP_COLOURS)], lw=lw)
    E = grid["ND"] + 1.5
    ax.set_xlim(-E, E); ax.set_ylim(-E, E); ax.set_aspect("equal"); ax.axis("off")
    return fig


def render_fill(grid, loops, drawn, width_px=2400, rings=None, petals=False, tip=1.25, G=None,
                level_colours=None):
    """Step 18. White lines; each dot's enclosed region filled by its drawn level.

    rings:  optional set of dots to circle (e.g. dots on diffraction orders).
    petals: sharpen every edge drop into an outward-pointing petal (needs G).
    Returns a PIL image.
    """
    ND = grid["ND"]
    cols = {k: tuple(int(c[i:i + 2], 16) for i in (1, 3, 5)) for k, c in (level_colours or LEVEL_COLOURS).items()}
    apex = {}
    if petals:
        for d in grid["dots"]:
            if any(g in grid["rim"] for g in gates_of(d)) and drawn[d] == 1:
                go = next(g for g in gates_of(d) if G.get(g, 0) == 1)
                opp = (2 * d[0] - go[0], 2 * d[1] - go[1])
                raw = (round(d[0] + .5 * (opp[0] - d[0]), 3), round(d[1] + .5 * (opp[1] - d[1]), 3))
                apex[raw] = (d[0] + tip * (opp[0] - d[0]), d[1] + tip * (opp[1] - d[1]))
    curves = []
    for lp in loops:
        P = []
        for p in lp["pts"]:
            key = (round(p[0], 3), round(p[1], 3))
            P += [apex[key]] * 2 if key in apex else [tuple(p)]    # doubled vertex stays sharp
        P = np.array(P)
        curves.append(chaikin(np.column_stack([(P[:, 0] + P[:, 1]) / 2, (P[:, 0] - P[:, 1]) / 2])))
    E = ND + (2.5 if petals else 1.5); W = width_px; s = W / (2 * E)
    px = lambda x, y: ((x + E) * s, (E - y) * s)
    mask = Image.new("L", (W, W), 0); dr = ImageDraw.Draw(mask)
    for S in curves:
        dr.line([px(a, b) for a, b in S], fill=255, width=max(3, int(.10 * s)), joint="curve")
    M = np.asarray(mask) > 0
    lab, _ = ndimage.label(~M)
    out = np.zeros((W, W, 3), np.uint8); out[:] = (20, 16, 12)
    for d in grid["dots"]:
        X, Y = px(*to_draw(d)); r = lab[int(Y), int(X)]
        if r:
            out[lab == r] = cols[max(1, drawn[d])]
    out[M] = (243, 234, 214)
    im = Image.fromarray(out)
    if rings:
        d2 = ImageDraw.Draw(im); R = .62 * s
        for d in rings:
            X, Y = px(*to_draw(d))
            d2.ellipse([X - R, Y - R, X + R, Y + R], outline=(53, 224, 192), width=max(5, int(.13 * s)))
    return im


# ============================================================================
# ALTERNATIVE GATE ASSIGNMENT: grow from the brightest dots downwards
# ============================================================================
def neighbours(grid, d):
    """Dots that share a gate with d."""
    return {o for g in gates_of(d) for o in grid["owner"][g] if o != d}


def level_growth(grid, want, seed=0, symmetric=False):
    """All gates start closed. Levels are handled from 4 down to 1.

    Within a level, dots touching an already-fixed dot go first (growing outward
    from what is fixed); when none are left, an isolated dot is picked at random.
    A dot opens as many of its still-free gates as it needs, chosen at random.
    Gates touched by a fixed dot never change again; rim gates stay closed.
    symmetric=True makes every choice in mirror groups, so the result is symmetric.
    Dots that cannot reach their level are left as they are.
    """
    rng = np.random.default_rng(seed)
    G = {g: 0 for g in grid["owner"]}
    fixed = set(grid["rim"])
    done, order = set(), []
    grp = mirror_group if symmetric else (lambda p: {p})

    def process(d):
        gs = gates_of(d)
        need = want[d] - sum(G[g] for g in gs)
        groups = {}
        for g in (g for g in gs if g not in fixed):
            groups.setdefault(min(grp(g)), []).append(g)
        keys, best = list(groups), []
        for r in range(len(keys) + 1):
            for sub in itertools.combinations(keys, r):
                got = sum(len(groups[k]) for k in sub)
                score = (abs(got - need), got > need)
                if not best or score < best[0][0]:
                    best = [(score, sub)]
                elif score == best[0][0]:
                    best.append((score, sub))
        chosen = best[rng.integers(len(best))][1] if best else ()
        for k in keys:
            for h in grp(k):
                G[h] = 1 if k in chosen else 0
                fixed.add(h)
        for e in grp(d):
            done.add(e)
        order.append(d)

    for L in (4, 3, 2, 1):
        todo = {d for d in grid["dots"] if want[d] == L}
        while todo - done:
            left = sorted(todo - done)
            front = [d for d in left if neighbours(grid, d) & done]
            if front:
                for d in rng.permutation(len(front)):
                    if front[d] not in done:
                        process(front[d])
            else:
                process(left[rng.integers(len(left))])
    drawn = dot_levels(grid, G)
    return dict(want=want, G=G, drawn=drawn, order=order,
                wrong=[d for d in grid["dots"] if drawn[d] != want[d]],
                sealed=[d for d in grid["dots"] if drawn[d] == 0])


def trace_everything(grid, G, seed=0):
    """trace_all, plus the sealed circles around dots with no open gate."""
    ND = grid["ND"]
    loops = trace_all(grid, G, seed=seed)
    A = gate_matrix(G, ND)
    seen = {frozenset(map(tuple, np.round(l["pts"], 3))) for l in loops}
    for d in grid["dots"]:
        if any(G.get(g, 0) for g in gates_of(d)):
            continue
        for g in gates_of(d):
            for h in range(4):
                st, pts, ok = (g[0], g[1], h), [], False
                for _ in range(8):                          # a circle closes in 4 moves
                    ni, nj, nh, pi, pj = next_step(A, ND, *st)
                    pts.append((pi, pj)); st = (ni, nj, nh)
                    if st == (g[0], g[1], h):
                        ok = True; break
                if ok and all(np.hypot(p[0] - d[0], p[1] - d[1]) < 1.01 for p in pts):
                    key = frozenset(map(tuple, np.round(pts, 3)))
                    if key not in seen:
                        seen.add(key)
                        loops.append(dict(start=(g[0], g[1], h), states=[], pts=np.array(pts)))
                    break
            else:
                continue
            break
    return loops


# ============================================================================
# HIGHLIGHTED RENDERING: make the bright dots (levels 3 and 4) stand out
# ============================================================================
def extent(grid, petals=False):
    """Half-width of the drawing: furthest dot from the centre plus a margin."""
    m = max(max(abs(x), abs(y)) for x, y in map(to_draw, grid["dots"]))
    return m + (2.5 if petals else 1.5)


def raster_regions(grid, loops, drawn, width_px=2400, petals=False, tip=1.25, G=None):
    """Rasterise the kolam. Returns (line_mask, level_map): level_map holds each pixel's
    dot level (1-4), 0 for background and for gaps that belong to no dot."""
    ND = grid["ND"]
    apex = {}
    if petals:
        for d in grid["dots"]:
            if any(g in grid["rim"] for g in gates_of(d)) and drawn[d] == 1:
                go = next(g for g in gates_of(d) if G.get(g, 0) == 1)
                opp = (2 * d[0] - go[0], 2 * d[1] - go[1])
                raw = (round(d[0] + .5 * (opp[0] - d[0]), 3), round(d[1] + .5 * (opp[1] - d[1]), 3))
                apex[raw] = (d[0] + tip * (opp[0] - d[0]), d[1] + tip * (opp[1] - d[1]))
    E = extent(grid, petals); W = width_px; s = W / (2 * E)
    px = lambda x, y: ((x + E) * s, (E - y) * s)
    mask = Image.new("L", (W, W), 0); dr = ImageDraw.Draw(mask)
    for lp in loops:
        P = []
        for p in lp["pts"]:
            key = (round(p[0], 3), round(p[1], 3))
            P += [apex[key]] * 2 if key in apex else [tuple(p)]
        P = np.array(P)
        S = chaikin(np.column_stack([(P[:, 0] + P[:, 1]) / 2, (P[:, 0] - P[:, 1]) / 2]))
        dr.line([px(a, b) for a, b in S], fill=255, width=max(3, int(.10 * s)), joint="curve")
    M = np.asarray(mask) > 0
    lab, n = ndimage.label(~M)
    lut = np.zeros(n + 1, np.int8)
    for d in grid["dots"]:
        X, Y = px(*to_draw(d)); r = lab[int(Y), int(X)]
        if r:
            lut[r] = max(1, drawn[d])
    return M, lut[lab], s


PALETTES = {
    # name: background, line, levels 1-4, glow for 4 and 3        (hex)
    "tu delft cyan":  dict(bg="#00A6D6", line="#FFFFFF", lv=["#0093C2", "#007DB0", "#EC6842", "#FFB81C"], g4="#FFE08A", g3="#F7A07E"),
    "tu delft navy":  dict(bg="#0C2340", line="#E8EEF5", lv=["#1B3A63", "#28507F", "#EC6842", "#FFB81C"], g4="#FFC94D", g3="#EC6842"),
    "warm white":     dict(bg="#FAF7F0", line="#3A2E2A", lv=["#E6DDEE", "#D8C3D6", "#E0663A", "#F2A900"], g4="#F2C14E", g3="#EFA07F"),
    "maroon":         dict(bg="#5C1A1B", line="#F6E7D0", lv=["#712829", "#873535", "#F28C28", "#FFD34E"], g4="#FFD34E", g3="#F28C28"),
    "red oxide floor":dict(bg="#8E3B26", line="#FFF8EC", lv=["#9D4731", "#AE553B", "#F2B134", "#FFF1C2"], g4="#FFF1C2", g3="#F2B134"),
    "purple":         dict(bg="#3B1F4F", line="#F3E9F7", lv=["#4A2A62", "#5C3676", "#EF60A3", "#FFB81C"], g4="#FFC94D", g3="#EF60A3"),
}


def _rgb(h):
    return np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)], float)


def render_palette(grid, loops, drawn, palette, width_px=2400, glow=True, dots=False,
                   dot_colour="#FFFFFF", dot_size=0.13, transparent=False, **kw):
    """Muted rendering in a named (or custom) palette: levels 1-2 a quiet step off the
    background, levels 3-4 in strong colour, with a soft glow around 4s and 3s."""
    P = PALETTES[palette] if isinstance(palette, str) else palette
    M, lev, s = raster_regions(grid, loops, drawn, width_px, **kw)
    out = np.zeros(M.shape + (3,)); out[:] = _rgb(P["bg"])
    for k in (1, 2, 3, 4):
        out[lev == k] = _rgb(P["lv"][k - 1])
    empty = (lev == 0) & ~M
    alpha = np.where(empty, 0.0, 1.0)
    if glow:
        for k, col, sig, amp in [(4, P["g4"], .55, .6), (3, P["g3"], .35, .28)]:
            if not (lev == k).any():
                continue
            h = gaussian_filter((lev == k).astype(float), sig * s)
            h = np.clip(h / max(h.max(), 1e-9) * 1.6, 0, 1) * amp
            c = _rgb(col)
            if transparent:                  # glow becomes a semi-transparent coloured halo
                a0 = alpha[empty]; add = h[empty] * (1 - a0)
                out[empty] = (out[empty] * a0[:, None] + c * add[:, None]) / np.maximum(a0 + add, 1e-9)[:, None]
                alpha[empty] = a0 + add
            else:
                out[empty] = out[empty] + h[empty][:, None] * (c - out[empty])
    out[M] = _rgb(P["line"])
    rgb = np.clip(out, 0, 255).astype(np.uint8)
    im = (Image.fromarray(np.dstack([rgb, (alpha * 255).astype(np.uint8)]), "RGBA") if transparent
          else Image.fromarray(rgb))
    if dots:                                    # the pulli, one at every dot
        E = extent(grid, kw.get("petals"))
        dr, r = ImageDraw.Draw(im), dot_size * s
        c = tuple(int(v) for v in _rgb(dot_colour))
        for d in grid["dots"]:
            x, y = to_draw(d); X, Y = (x + E) * s, (E - y) * s
            dr.ellipse([X - r, Y - r, X + r, Y + r], fill=c)
    return im


def render_highlight(grid, loops, drawn, width_px=2400, style="muted", glow=True, **kw):
    """style "muted": levels 1-2 in dark dusky tones, 3-4 vivid.
    style "spotlight": only levels 3-4 filled; lines bright only where they border them.
    glow: soft gold halo around level-4 dots (and a faint one around level 3)."""
    M, lev, s = raster_regions(grid, loops, drawn, width_px, **kw)
    bg = np.array([20, 16, 12], float)
    out = np.zeros(M.shape + (3,)); out[:] = bg
    if style == "muted":
        pal = {1: (44, 36, 72), 2: (78, 30, 56), 3: (247, 128, 26), 4: (255, 206, 42)}
    else:
        pal = {1: tuple(bg), 2: tuple(bg), 3: (247, 128, 26), 4: (255, 206, 42)}
    for k, c in pal.items():
        out[lev == k] = c
    if glow:
        for k, col, sig, amp in [(4, (255, 200, 60), .55, 1.0), (3, (247, 128, 26), .35, .45)]:
            h = gaussian_filter((lev == k).astype(float), sig * s)
            h = np.clip(h / max(h.max(), 1e-9) * 1.6, 0, 1) * amp
            empty = (lev == 0) & ~M if style == "muted" else (lev <= 2) & ~M
            for c in range(3):
                out[..., c] = np.where(empty, out[..., c] + h * (col[c] - out[..., c]) * .55, out[..., c])
    cream = np.array([243, 234, 214], float)
    if style == "spotlight":
        near = ndimage.binary_dilation(lev >= 3, iterations=max(2, int(.12 * s)))
        out[M & near] = cream
        out[M & ~near] = (92, 84, 74)
    else:
        out[M] = cream
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))


# ============================================================================
# OUTLINES OTHER THAN THE DIAMOND
# ============================================================================
def build_grid_shape(ND, keep):
    """Like build_grid, but only dots whose sample (u, v) satisfies keep(u, v).

    Dots with no kept neighbour are dropped, then only the largest connected piece is
    kept. Gates on the new outline get a single owner, so they become the rim."""
    full = build_grid(ND)
    dots = {d for d in full["dots"] if keep(*full["uv"][d])}
    nb = lambda d: {o for g in gates_of(d) for o in full["owner"][g] if o != d and o in dots}
    left, best = set(dots), set()
    while left:                                   # largest 4-connected component
        seed = left.pop(); comp, stack = {seed}, [seed]
        while stack:
            for o in nb(stack.pop()):
                if o in left:
                    left.discard(o); comp.add(o); stack.append(o)
        best = max(best, comp, key=len)
    dots = sorted(best)
    owner = {}
    for d in dots:
        for g in gates_of(d):
            owner.setdefault(g, []).append(d)
    uv = {d: full["uv"][d] for d in dots}
    return dict(ND=ND, R=full["R"], dots=dots, uv=uv, owner=owner,
                rim={g for g, o in owner.items() if len(o) == 1},
                shell={d: abs(uv[d][0]) + abs(uv[d][1]) for d in dots})


def _polar(u, v):
    return np.hypot(u, v), np.arctan2(v, u)


def mandelbrot_keep(scale=0.062, centre=-0.35, iters=80):
    """Dots whose sample maps into the Mandelbrot set; the real axis runs along u."""
    def keep(u, v):
        c = complex(centre + scale * u, scale * v); z = 0j
        for _ in range(iters):
            z = z * z + c
            if abs(z) > 2:
                return False
        return True
    return keep


SHAPES = {
    "diamond":         lambda R: (lambda u, v: abs(u) + abs(v) <= R),
    "circle":          lambda R: (lambda u, v: u * u + v * v <= R * R),
    "quatrefoil":      lambda R: (lambda u, v: _polar(u, v)[0] <= R * (0.62 + 0.38 * abs(np.cos(2 * _polar(u, v)[1])))),
    "four-point star": lambda R: (lambda u, v: (abs(u) ** 0.6 + abs(v) ** 0.6) ** (1 / 0.6) <= R),
    "eight-point star":lambda R: (lambda u, v: _polar(u, v)[0] <= R * (0.5 + 0.5 * abs(np.cos(2 * _polar(u, v)[1])) ** 3
                                                           + 0.0) if False else
                                   _polar(u, v)[0] <= R * (0.48 + 0.52 * abs(np.cos(4 * _polar(u, v)[1])) ** 2)),
}


# ============================================================================
# ARBITRARY SPACING, MASKS, ONE-CALL BUILD, EXPORT   (used by kolam_studio.py)
# ============================================================================
def sample_spacing(sp, du, dv, pts, lattice=None, tol=0.08):
    """Sample the spectrum at u*du, v*dv (cycles per image) for each (u, v) in pts.
    Points beyond Nyquist are dropped. on_lattice marks samples within tol of a
    diffraction order of the given lattice."""
    cc, up, half = sp["cc"], sp["up"], sp["native"] / 2 - 1
    pts = [p for p in pts if abs(p[0] * du) < half and abs(p[1] * dv) < half]
    pix = {p: (int(round(cc + p[1] * dv * up)), int(round(cc + p[0] * du * up))) for p in pts}
    if lattice:
        a, b = lattice["a_star_native"], lattice["b_star_native"]
        near = lambda x: abs(x - round(x)) < tol
        on = {p: near(p[0] * du / a) and near(p[1] * dv / b) for p in pts}
    else:
        on = {p: False for p in pts}
    return dict(pts=pts, du=du, dv=dv, pix=pix, on_lattice=on,
                logamp={p: float(np.log10(sp["amp"][pix[p]] + 1e-12)) for p in pts},
                phase={p: float(np.angle(sp["F"][pix[p]])) for p in pts})


def make_levels_spacing(sp, du, dv, pts, lattice=None):
    """Sample, symmetrise, normalise, quantise at an arbitrary spacing."""
    S = sample_spacing(sp, du, dv, pts, lattice)
    S["sym"] = symmetrise(S["logamp"])
    S["norm"] = normalise(S["sym"], exclude_dc=(0, 0) in S["sym"])
    S["level"] = quantise(S["norm"])
    return S


def shape_keep(name, R):
    """A predefined outline of half-width R (in samples) as a keep(u, v) test."""
    if name == "mandelbrot":
        return mandelbrot_keep(scale=1.2 / R)
    return SHAPES[name](R)


def mask_keep(mask, R, invert=False, threshold=0.5):
    """Use an image as the outline: it covers u, v in [-R, R]; dark pixels are inside."""
    a = np.asarray(mask.convert("L"), float) / 255.0
    h, w = a.shape
    def keep(u, v):
        x = int(round((u + R) / (2 * R) * (w - 1))); y = int(round((R - v) / (2 * R) * (h - 1)))
        if not (0 <= x < w and 0 <= y < h):
            return False
        inside = a[y, x] < threshold
        return inside != invert
    return keep


def is_symmetric(grid):
    """True if the dot set is symmetric under both mirrors (needed for mirror groups)."""
    D = set(grid["dots"])
    return all(e in D for d in D for e in mirror_group(d))


def build_kolam(sp, du, dv, keep, R, lattice=None, method="growth", seed=2):
    """Everything from sampling to traced loops, for any spacing and outline.

    method "growth": level growth (4s first); "sweep": shell sweep from DC.
    Mirror groups are used whenever the outline allows it. Sealed dots are rescued."""
    half = sp["native"] / 2 - 1
    ok = lambda u, v: keep(u, v) and abs(u * du) < half and abs(v * dv) < half
    ND = 4 * int(np.ceil(1.5 * R)) + 1
    g = build_grid_shape(ND, ok)
    S = make_levels_spacing(sp, du, dv, [g["uv"][d] for d in g["dots"]], lattice)
    want = {d: S["level"][g["uv"][d]] for d in g["dots"]}
    sym = is_symmetric(g)
    if method == "sweep" and sym:
        G, _ = shell_sweep(g, want, close_rim(g))
    else:
        G = level_growth(g, want, seed=seed, symmetric=sym)["G"]
    G, rescued = rescue_sealed(g, G, want=want, symmetric=sym)
    drawn = dot_levels(g, G)
    loops = trace_everything(g, G, seed=1)
    tot = 4 * len(g["dots"])
    orders = [d for d in g["dots"] if S["on_lattice"][g["uv"][d]]]
    stats = dict(dots=len(g["dots"]), symmetric=sym, loops=len(loops),
                 longest=max(len(l["pts"]) for l in loops) / tot,
                 wrong=sum(drawn[d] != want[d] for d in g["dots"]),
                 orders=len(orders), orders_bright=sum(drawn[d] >= 3 for d in orders),
                 method=("sweep" if method == "sweep" and sym else "growth"))
    return dict(grid=g, S=S, want=want, G=G, drawn=drawn, loops=loops, stats=stats)


def render_kolam(K, colours, width_px=1600, dots=True, petals=True, glow=True, transparent=False,
                 dot_size=0.13, tip=1.25, margin=0.03):
    """Render a built kolam and crop it to a square around its content.
    colours: dict with bg, line, lv (four hex colours), g4, g3, dot."""
    pal = dict(bg=colours["bg"], line=colours["line"], lv=list(colours["lv"]),
               g4=colours.get("g4", colours["lv"][3]), g3=colours.get("g3", colours["lv"][2]))
    im = render_palette(K["grid"], K["loops"], K["drawn"], pal, width_px=width_px, glow=glow,
                        dots=dots, dot_colour=colours.get("dot", "#FFFFFF"), dot_size=dot_size,
                        transparent=transparent, petals=petals, tip=tip, G=K["G"])
    a = np.asarray(im)
    m = a[..., 3] > 8 if transparent else np.abs(a.astype(int) - _rgb(pal["bg"])).sum(2) > 12
    ys, xs = np.nonzero(m)
    cx, cy = (xs.min() + xs.max()) // 2, (ys.min() + ys.max()) // 2
    h = int(max(xs.max() - xs.min(), ys.max() - ys.min()) / 2 * (1 + margin)) + 2
    fill = (0, 0, 0, 0) if transparent else tuple(int(v) for v in _rgb(pal["bg"]))
    big = Image.new(im.mode, (im.width + 4 * h, im.height + 4 * h), fill); big.paste(im, (2 * h, 2 * h))
    return big.crop((cx + h, cy + h, cx + 3 * h, cy + 3 * h))


def export_image(im, path, fmt="pdf", width_mm=118.0, dpi=600, flatten_bg="#FFFFFF"):
    """Save a square kolam image at a physical size.

    pdf: vector page of width_mm with the image embedded (transparency kept).
    png: pixels = width_mm at dpi, dpi stored in the file (transparency kept).
    eps: EPS cannot hold transparency; a transparent image is flattened on flatten_bg.
    Returns (path, message)."""
    import io
    fmt = fmt.lower(); px = int(round(width_mm / 25.4 * dpi))
    im = im.resize((px, px), Image.LANCZOS)
    note = f"{px} x {px} px, {width_mm:g} mm at {dpi} dpi"
    if fmt == "png":
        im.save(path, dpi=(dpi, dpi))
    elif fmt == "pdf":
        from reportlab.pdfgen import canvas
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        buf = io.BytesIO(); im.save(buf, "PNG"); buf.seek(0)
        c = canvas.Canvas(path, pagesize=(width_mm * mm, width_mm * mm))
        c.drawImage(ImageReader(buf), 0, 0, width_mm * mm, width_mm * mm, mask="auto")
        c.showPage(); c.save()
    elif fmt == "eps":
        if im.mode == "RGBA":
            base = Image.new("RGB", im.size, tuple(int(v) for v in _rgb(flatten_bg)))
            base.paste(im, mask=im.getchannel("A")); im = base
            note += "; EPS has no transparency, flattened on " + flatten_bg
        buf = io.BytesIO(); im.convert("RGB").save(buf, "EPS"); data = buf.getvalue()
        pt = width_mm / 25.4 * 72
        data = data.replace(f"%%BoundingBox: 0 0 {px} {px}".encode(),
                            f"%%BoundingBox: 0 0 {int(round(pt))} {int(round(pt))}\n%%HiResBoundingBox: 0 0 {pt:.3f} {pt:.3f}".encode(), 1)
        data = data.replace(f"{px} {px} scale".encode(), f"{pt:.3f} {pt:.3f} scale".encode(), 1)
        open(path, "wb").write(data)
    else:
        raise ValueError("format must be pdf, png or eps")
    return path, note


# the six outlines as delivered in red oxide (sample units; spacing = lattice / 3)
PRESETS = {"diamond": 20, "circle": 16.28, "quatrefoil": 18.82,
           "four-point star": 29.13, "eight-point star": 21.39, "mandelbrot": 1.2 / 0.044}
