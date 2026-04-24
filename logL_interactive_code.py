import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

# ---------------------------------------------------------------------------
# Parameters
#   Website  : read reactively from the OJS form via the pyodide JS proxy.
#   Standalone: parsed from the command line (or IPython's %run args).
#               Example: python logL_interactive_code.py --y1 3 --b2 0.1
#               Colorbar limits are auto-computed unless --lmin / --lmax given.
# ---------------------------------------------------------------------------
try:
    _p = {
        k: params[k]
        for k in (  # pyodide: OJS reactive proxy
            "y1",
            "y2",
            "b1",
            "b2",
            "xmin",
            "xmax",
            "x1_start",
            "x2_start",
        )
    }
    _lmin = _lmax = None  # always auto-compute on the website
except NameError:
    import argparse

    _parser = argparse.ArgumentParser(
        description="Log-likelihood landscape and MLEM / MLEM-3 trajectories "
        "for the 2-voxel Poisson problem."
    )
    _parser.add_argument(
        "--y1",
        type=int,
        default=1,
        metavar="N",
        help="measured counts bin 1 (default: 1)",
    )
    _parser.add_argument(
        "--y2",
        type=int,
        default=2,
        metavar="N",
        help="measured counts bin 2 (default: 2)",
    )
    _parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        metavar="F",
        help="background bin 1 (default: 0.5)",
    )
    _parser.add_argument(
        "--b2",
        type=float,
        default=0.5,
        metavar="F",
        help="background bin 2 (default: 0.5)",
    )
    _parser.add_argument(
        "--xmin",
        type=float,
        default=-0.5,
        metavar="F",
        help="axis min for x1 and x2 (default: -0.5)",
    )
    _parser.add_argument(
        "--xmax",
        type=float,
        default=5.0,
        metavar="F",
        help="axis max for x1 and x2 (default: 5.0)",
    )
    _parser.add_argument(
        "--lmin",
        type=float,
        default=None,
        metavar="F",
        help="colorbar min (default: auto)",
    )
    _parser.add_argument(
        "--lmax",
        type=float,
        default=None,
        metavar="F",
        help="colorbar max (default: auto)",
    )
    _parser.add_argument(
        "--x1_start",
        type=float,
        default=2.5,
        metavar="F",
        help="MLEM start x1 (default: 2.5)",
    )
    _parser.add_argument(
        "--x2_start",
        type=float,
        default=2.5,
        metavar="F",
        help="MLEM start x2 (default: 2.5)",
    )
    _args = _parser.parse_args()
    _p = {
        k: getattr(_args, k)
        for k in ("y1", "y2", "b1", "b2", "xmin", "xmax", "x1_start", "x2_start")
    }
    _lmin = _args.lmin
    _lmax = _args.lmax

y1 = int(_p["y1"])
y2 = int(_p["y2"])
b1 = float(_p["b1"])
b2 = float(_p["b2"])
xmin = float(_p["xmin"])
xmax = float(_p["xmax"])
x1_start = float(_p["x1_start"])
x2_start = float(_p["x2_start"])

# Guard against inverted axis range
xmin, xmax = min(xmin, xmax - 0.5), max(xmax, xmin + 0.5)


# ---------------------------------------------------------------------------
# Log-likelihood (vectorised for grid, scalar for trajectory)
# Forward model: ȳ_1 = x_1 + b_1,  ȳ_2 = x_1 + x_2 + b_2
# ---------------------------------------------------------------------------
def logL(x1, x2):
    """Vectorised log-likelihood on a meshgrid."""
    ybar1 = x1 + b1
    ybar2 = x1 + x2 + b2
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(
            y1 > 0,
            y1 * np.log(np.where(ybar1 > 0, ybar1, np.nan)) - ybar1,
            np.where(ybar1 > 0, -ybar1, -np.inf),
        )
        t2 = np.where(
            y2 > 0,
            y2 * np.log(np.where(ybar2 > 0, ybar2, np.nan)) - ybar2,
            np.where(ybar2 > 0, -ybar2, -np.inf),
        )
    return t1 + t2


def logL_scalar(x1, x2):
    """Scalar log-likelihood at a single point."""
    ybar1 = x1 + b1
    ybar2 = x1 + x2 + b2
    if ybar1 <= 0 or ybar2 <= 0:
        return -np.inf
    t1 = y1 * np.log(ybar1) - ybar1 if y1 > 0 else -ybar1
    t2 = y2 * np.log(ybar2) - ybar2 if y2 > 0 else -ybar2
    return t1 + t2


# ---------------------------------------------------------------------------
# Gradient of L w.r.t. (x1, x2)
# ∂L/∂x_j = Σ_i a_ij (y_i / ȳ_i − 1)   [sensitivity s_j = Σ_i a_ij]
# Voxel 1: s_1 = 2,  ∂L/∂x_1 = (y1/ȳ1 − 1) + (y2/ȳ2 − 1)
# Voxel 2: s_2 = 1,  ∂L/∂x_2 = (y2/ȳ2 − 1)
# ---------------------------------------------------------------------------
def grad_L(x1, x2):
    ybar1 = x1 + b1
    ybar2 = x1 + x2 + b2
    r1 = y1 / ybar1 if ybar1 > 0 else 0.0
    r2 = y2 / ybar2 if ybar2 > 0 else 0.0
    return (r1 - 1) + (r2 - 1), (r2 - 1)


# ---------------------------------------------------------------------------
# Update rules as preconditioned gradient ascent
# gam=0  →  standard MLEM
# gam>0  →  MLEM-3 (Fessler eq. 18.4.5)
# ---------------------------------------------------------------------------
def mlem_update(x1, x2, gam=0.0):
    """x_j ← [x_j + ((x_j + γ) / s_j) · ∂L/∂x_j]₊"""
    g1, g2 = grad_L(x1, x2)
    return (
        max(0.0, x1 + (x1 + gam) / 2.0 * g1),
        max(0.0, x2 + (x2 + gam) / 1.0 * g2),
    )


# γ_j = min_i  r̄_i / a_{i·}   (Fessler eq. 18.4.6, background as surrogate for r̄_i)
# bin 1: a_{1·} = 1,  bin 2: a_{2·} = 2  →  γ = min(b1/1, b2/2)
gamma = min(b1 / 1.0, b2 / 2.0)

# ---------------------------------------------------------------------------
# Contour grid
# ---------------------------------------------------------------------------
x = np.linspace(xmin, xmax, 800)
X1g, X2g = np.meshgrid(x, x)
Z = logL(X1g, X2g)

# Contours only where ȳ_i > 0
defined = (X1g + b1 > 0) & (X1g + X2g + b2 > 0)
Z_plot = np.where(defined, Z, np.nan)

# Constrained maximum (grid search over non-negative orthant)
Z_feas = np.where((X1g >= 0) & (X2g >= 0), Z_plot, np.nan)
idx = np.unravel_index(np.nanargmax(Z_feas), Z_feas.shape)
xhat1, xhat2 = float(X1g[idx]), float(X2g[idx])

# Unconstrained maximum
xml1 = y1 - b1
xml2 = y2 - xml1 - b2

# ---------------------------------------------------------------------------
# Colorbar limits
#   lmax = L at the unconstrained maximum (highest value of the objective)
#   lmin = minimum finite L in the displayed grid region
#   Either can be overridden via --lmin / --lmax on the command line.
# ---------------------------------------------------------------------------
_finite = Z_plot[np.isfinite(Z_plot)]

lmax_auto = logL_scalar(xml1, xml2)
if not np.isfinite(lmax_auto):  # unconstrained max outside grid / undefined
    lmax_auto = float(_finite.max()) if _finite.size > 0 else 0.0

lmin_auto = float(_finite.min()) if _finite.size > 0 else lmax_auto - 20.0

lmin = lmin_auto if _lmin is None else _lmin
lmax = lmax_auto if _lmax is None else _lmax
if lmax <= lmin:  # guard against degenerate override
    lmax = lmin + 1.0

levels = np.linspace(lmin, lmax, 40)

# ---------------------------------------------------------------------------
# Trajectories
# ---------------------------------------------------------------------------
N_ITER = 20
tx, ty = [x1_start], [x2_start]
tx3, ty3 = [x1_start], [x2_start]
for _ in range(N_ITER):
    nx, ny = mlem_update(tx[-1], ty[-1], gam=0.0)
    nx3, ny3 = mlem_update(tx3[-1], ty3[-1], gam=gamma)
    tx.append(nx)
    ty.append(ny)
    tx3.append(nx3)
    ty3.append(ny3)

iters = np.arange(N_ITER + 1)

# Log-likelihood along trajectories
L_mlem = [logL_scalar(x_, y_) for x_, y_ in zip(tx, ty)]
L_mlem3 = [logL_scalar(x_, y_) for x_, y_ in zip(tx3, ty3)]

# Distance to constrained maximum
d_mlem = [np.sqrt((x_ - xhat1) ** 2 + (y_ - xhat2) ** 2) for x_, y_ in zip(tx, ty)]
d_mlem3 = [np.sqrt((x_ - xhat1) ** 2 + (y_ - xhat2) ** 2) for x_, y_ in zip(tx3, ty3)]

# ---------------------------------------------------------------------------
# Figure: 3-row layout — contour map, L(x) vs iter, ||x−x̂|| vs iter
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(6, 6), layout="constrained")
ax_map = ax[0, 1]
ax_L = ax[1, 0]
ax_dist = ax[1, 1]
ax[0, 0].set_visible(False)

# — Contour map —
cs = ax_map.contourf(
    X1g, X2g, Z_plot, levels=levels, cmap="magma", alpha=0.85, vmin=lmin, vmax=lmax
)
ax_map.axhline(0, color="black", lw=1.0)
ax_map.axvline(0, color="black", lw=1.0)
ax_map.plot(tx, ty, "k.-", ms=5, zorder=5, label="MLEM")
ax_map.plot(tx3, ty3, "c.-", ms=5, zorder=5, label="MLEM-3")
ax_map.plot(tx[0], ty[0], "ks", ms=7, zorder=6, label="start")
ax_map.plot(xhat1, xhat2, "b^", ms=9, zorder=6)
ax_map.plot(xml1, xml2, "r*", ms=11, zorder=6)
cb = plt.colorbar(cs, ax=ax_map, fraction=0.04, pad=0.01)
cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
ax_map.set_xlabel(r"$x_1$")
ax_map.set_ylabel(r"$x_2$")
ax_map.set_xlim(xmin, xmax)
ax_map.set_ylim(xmin, xmax)
ax_map.set_aspect("equal", adjustable="box")
ax_map.set_title(
    rf"$y_1={y1},\; y_2={y2},\; b_1={b1:.1f},\; b_2={b2:.1f}$", fontsize="small"
)
ax_map.legend(loc="upper right", fontsize=7.5)

# — L(x) vs iteration —
ax_L.plot(iters, L_mlem, "k.-", ms=4, label="MLEM")
ax_L.plot(iters, L_mlem3, "c.-", ms=4, label="MLEM-3")
Lmax = logL_scalar(xhat1, xhat2)
ax_L.axhline(Lmax, color="b", lw=1, ls="--", label=r"$\mathcal{L}(\hat{x})$")
ax_L.set_xlim(-1, None)
ax_L.set_ylim(L_mlem[1], Lmax + 0.05 * (Lmax - L_mlem[1]))
ax_L.set_xlabel("iteration $k$")
ax_L.set_ylabel(r"$\mathcal{L}(x^k)$")
ax_L.set_title("Log-likelihood", fontsize="medium")
ax_L.legend(fontsize=7.5)
ax_L.grid(":")

# — ||x − x̂|| vs iteration —
ax_dist.loglog(iters, d_mlem, "k.-", ms=4, label="MLEM")
ax_dist.loglog(iters, d_mlem3, "c.-", ms=4, label="MLEM-3")
ax_dist.set_xlabel("iteration $k$")
ax_dist.set_ylabel(r"$\|x^k - \hat{x}\|$")
ax_dist.set_title(r"Distance to $\hat{x}$", fontsize="medium")
ax_dist.legend(fontsize=7.5)
ax_dist.grid(":")

plt.show()
