"""
Closed-form counterexample of Section 3.

X ~ Exp(1) (so V = 1) and g(x) = (1+k) e^{-kx}, k > 0, which is decreasing and
satisfies E g(X) = 1.  S^g_inf < S^hom_inf exactly when

    k > k*(R0) := (1/S^hom_inf - 1)/(R0 - 1) - 1,

so that heterogeneity may increase the epidemic size
"""
import numpy as np
from scipy.optimize import brentq
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def tstar(R):
    """root of e^{-t} - 1 + t/R = 0."""
    return brentq(lambda t: -np.expm1(-t) - t / R, 0.5 * (R - 1), 12 * R, xtol=1e-15)


def Shom(R):
    """homogeneous final susceptible fraction, = exp(-t*(R))."""
    return np.exp(-tstar(R))


def Sg(k, R):
    """final susceptible fraction of the closed-form family, eq. (32)."""
    return 1.0 / (1.0 + (1.0 + k) * (R - 1.0))


def kstar(R):
    """threshold of eq. (33)."""
    return (1.0 / Shom(R) - 1.0) / (R - 1.0) - 1.0





# also check sbar_inf solves the final size relation, as a sanity test
for R, k in [(2.0, 3.0), (3.0, 0.7)]:
    p = 1.0                       # X ~ Exp(1) = Gamma(1,1)
    m = 1.0 / (1.0 + k)
    sb = np.exp(-(1 + k) * (R - 1))
    lhs = (1 + k) / (1 + k - np.log(sb)) - np.log(sb) / ((1 + k) * R)
    print(f"  final size relation at R0={R}, k={k}: LHS = {lhs:.12f} (should be 1)")

# ----------------------------------------------------------------- figure
mpl.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
                     "legend.fontsize": 9, "figure.dpi": 130, "savefig.dpi": 130,
                     "mathtext.fontset": "dejavusans"})
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12.4, 4.9))

# ---- panel (a): S^g_inf against k, for three values of R0
R0S = [1.5, 2.0, 3.0]
COL = ["#1565c0", "#c62828", "#2e7d32"]
MRK = ["o", "s", "D"]

kk = np.linspace(0, 12, 800)
for R, c, mk in zip(R0S, COL, MRK):
    ax0.plot(kk, [Sg(k, R) for k in kk], color=c, lw=2.0, zorder=4)
    ax0.axhline(Shom(R), color=c, ls=":", lw=1.3, zorder=2)
    ks = kstar(R)
    ax0.plot([ks], [Shom(R)], color=c, marker=mk, ms=7.5, mec="white", mew=1.0,
             zorder=6)
    ax0.annotate(rf"$S^{{\mathrm{{hom}}}}_\infty={Shom(R):.3f}$",
                 xy=(0.15, Shom(R)), xytext=(0, 5), textcoords="offset points",
                 color=c, fontsize=8.4, ha="left")

ax0.set_xlim(0, 12)
ax0.set_ylim(0, 0.72)
ax0.set_xlabel(r"$k$")
ax0.set_ylabel(r"$S^{g}_{\infty}$")
ax0.set_title(r"(a)  $S^{g}_{\infty}=\left[1+(1+k)(\mathcal{R}_0-1)\right]^{-1}$")
ax0.grid(alpha=0.14, lw=0.6)
ax0.legend(handles=[Line2D([], [], color=c, lw=2.0, marker=mk, ms=7,
                           mec="white", mew=0.9,
                           label=rf"$\mathcal{{R}}_0={R:g}$")
                    for R, c, mk in zip(R0S, COL, MRK)],
           loc="upper right", framealpha=0.96)

# ---- panel (b): the threshold k*(R0)
RR = np.linspace(1.02, 6.0, 700)
KS = np.array([kstar(R) for R in RR])

ax1.fill_between(RR, KS, 30, color="#c62828", alpha=0.13, lw=0, zorder=0)
ax1.plot(RR, KS, color="#37474f", lw=2.3, zorder=4)
for R, c, mk in zip(R0S, COL, MRK):
    ax1.plot([R], [kstar(R)], color=c, marker=mk, ms=7.5, mec="white", mew=1.0,
             zorder=6)

ax1.annotate(r"$k>k^{*}$:  $S^{g}_{\infty}<S^{\mathrm{hom}}_{\infty}$," "\n"
             r"heterogeneity increases" "\n" "the epidemic size",
             xy=(1.15, 17.5), color="#8e2020", fontsize=8.6, ha="left",
             va="center")
ax1.annotate(r"$k<k^{*}$:  $S^{g}_{\infty}>S^{\mathrm{hom}}_{\infty}$",
             xy=(4.55, 3.0), color="0.35", fontsize=8.6, ha="center")

ax1.set_xlim(1.02, 6)
ax1.set_ylim(0, 22)
ax1.set_xlabel(r"$\mathcal{R}_0$")
ax1.set_ylabel(r"$k^{*}(\mathcal{R}_0)$")
ax1.set_title(r"(b)  threshold $k^{*}(\mathcal{R}_0)$ ")
ax1.grid(alpha=0.14, lw=0.6)

plt.show()