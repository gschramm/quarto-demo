import numpy as np
import matplotlib.pyplot as plt


def logL(x1, x2, y1, y2, b1, b2):
    ybar1 = x1 + b1
    ybar2 = x1 + x2 + b2
    with np.errstate(divide="ignore", invalid="ignore"):
        term1 = np.where(
            y1 > 0,
            y1 * np.log(np.where(ybar1 > 0, ybar1, np.nan)) - ybar1,
            np.where(ybar1 > 0, -ybar1, -np.inf),
        )
        term2 = np.where(
            y2 > 0,
            y2 * np.log(np.where(ybar2 > 0, ybar2, np.nan)) - ybar2,
            np.where(ybar2 > 0, -ybar2, -np.inf),
        )
    return term1 + term2


def mlem_update(x1, x2, y1, y2, b1, b2):
    ybar1 = x1 + b1
    ybar2 = x1 + x2 + b2

    r1 = np.where(ybar1 > 0, y1 / ybar1, 0)
    r2 = np.where(ybar2 > 0, y2 / ybar2, 0)

    x1_new = float((x1 / 2) * (r1 + r2))
    x2_new = float(x2 * r2)

    return x1_new, x2_new


def plot_logL(axx, y1, y2, b1, b2, X1, X2, show_grad=False):
    Z = logL(X1, X2, y1=y1, y2=y2, b1=b1, b2=b2)
    feasible = (X1 + b1 >= 0) & (X1 + X2 + b2 >= 0)
    constrain = (X1 >= 0) & (X2 >= 0)
    Z_plot = np.where(feasible, Z, np.nan)

    xml_unconstrained1 = y1 - b1
    xml_unconstrained2 = y2 - xml_unconstrained1 - b2

    Z_constrained = np.where(constrain, Z_plot, np.nan)
    flat_idx = np.nanargmax(Z_constrained)
    idx = np.unravel_index(flat_idx, Z_constrained.shape)
    xhat_1 = X1[idx]
    xhat_2 = X2[idx]

    Lmin = Z_plot[constrain].min()
    Lmax = Z_plot[feasible].max()

    levels = np.linspace(Lmin, Lmax, 32)
    cs = axx.contourf(X1, X2, Z_plot, levels=levels, cmap="magma", alpha=0.85)
    axx.hlines(0, 0, X1.max(), color="black", lw=1.0, ls="-")
    axx.vlines(0, 0, X2.max(), color="black", lw=1.0, ls="-")

    # show 20 MLEM iteraions starting from (2.5, 2.5) as a black line with dots
    x1_traj = [2.5]
    x2_traj = [2.5]
    for _ in range(20):
        x1_new, x2_new = mlem_update(x1_traj[-1], x2_traj[-1], y1, y2, b1, b2)
        x1_traj.append(x1_new)
        x2_traj.append(x2_new)
    axx.plot(x1_traj, x2_traj, "k.-", ms=5, zorder=5, label="MLEM trajectory")

    # showt the constrained maximum xhat as a blue rectangle
    axx.plot(
        xhat_1,
        xhat_2,
        "b^",
        ms=7,
        zorder=5,
    )

    # show the unconstrained maximum as a red star
    axx.plot(
        xml_unconstrained1,
        xml_unconstrained2,
        "r*",
        ms=7,
        zorder=5,
    )
    plt.colorbar(
        cs, ax=axx, label=r"$\log L$", fraction=0.04, location="right", pad=0.04
    )

    axx.set_xlabel(r"$x_1$")
    axx.set_ylabel(r"$x_2$")
    axx.set_aspect("equal", adjustable="box")

    if show_grad:
        # show the gradient at the constrained maximum
        grad_x1 = (y1 / (xhat_1 + b1) - 1) + (y2 / (xhat_1 + xhat_2 + b2) - 1)
        grad_x2 = y2 / (xhat_1 + xhat_2 + b2) - 1
        axx.quiver(
            xhat_1,
            xhat_2,
            grad_x1,
            grad_x2,
            scale_units="width",
            scale=10.0 * np.sqrt(grad_x1**2 + grad_x2**2),
            color="grey",
        )

    axx.set_title(f"$y_1={y1}$, $y_2={y2}$, $b_1={b1}$, $b_2={b2}$", fontsize="medium")


################################################################################
################################################################################
################################################################################


x1 = np.linspace(-1.2, 3.0, 1001)
x2 = np.linspace(-1.2, 3.0, 1001)
X1, X2 = np.meshgrid(x1, x2)

fig, ax = plt.subplots(1, 3, figsize=(14.25, 4), layout="constrained")

plot_logL(ax[0], y1=0, y2=2, b1=0.5, b2=0.5, X1=X1, X2=X2, show_grad=True)
plot_logL(ax[1], y1=1, y2=2, b1=0.5, b2=0.5, X1=X1, X2=X2)
plot_logL(ax[2], y1=3, y2=2, b1=0.5, b2=0.5, X1=X1, X2=X2, show_grad=True)

plt.show()
