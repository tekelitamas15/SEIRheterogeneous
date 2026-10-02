"""
Upper bound, two panels.

For every pair (P_X, g) with E(X)=E(Y)=1 and m = E(XY),

    A  <=  A^+ = 1 - exp(-t*(R0)/m_1)

Six susceptibility laws of mean 1 are shown, each multiplied with

    (a)  g(x) = c x^k,      k >= 0   (non-decreasing, m >= 1),
    (b)  g(x) = c e^{-kx},  k >  0   (non-increasing,  m <= 1).

"""
import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_jacobi
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

R0 = 2.0


def tstar(R):
    return brentq(lambda t: -np.expm1(-t) - t / R, 0.5 * (R - 1), 12 * R, xtol=1e-15)


TST = tstar(R0)
ZSTAR = TST / R0


# --------------------------------------------------------------- the six laws
class Gam:
    """Gamma with mean 1 and variance V: shape p = 1/V, rate p.  Exact."""
    b = np.inf

    def __init__(s, V, name, color, marker):
        s.p, s.V, s.name, s.color, s.marker = 1.0 / V, V, name, color, marker

    def data(s, k, case):
        p = s.p
        if case == "b":                      # g ~ x^k   -> mu = Gamma(p+k, p)
            shape, rate = p + k, p
        else:                                # g ~ e^-kx -> mu = Gamma(p, p+k)
            shape, rate = p, p + k
        m = shape / rate
        Dmu = lambda u: np.expm1(-shape * np.log1p(u / rate))
        Lph = lambda u: np.exp(-p * np.log1p(u / p))
        return m, Dmu, Lph


class Nod:
    """Atomic or quadrature law."""

    def __init__(s, x, pi, name, color, marker):
        s.x, s.pi = x, pi / pi.sum()
        s.name, s.color, s.marker = name, color, marker
        s.b = float(x.max())
        s.V = float(s.pi @ x**2 - (s.pi @ x) ** 2)

    def data(s, k, case):
        x, pi = s.x, s.pi
        if case == "b":
            lg = k * np.log(x / s.b)                      # overflow-safe x^k
        else:
            lg = -k * (x - x.min())                       # overflow-safe e^-kx
        lg -= lg.max()
        q = pi * np.exp(lg)
        q = q / q.sum()
        return (float(q @ x),
                lambda u: float(q @ np.expm1(-u * x)),
                lambda u: float(pi @ np.exp(-u * x)))


def beta_nodes(a, b, L, n=400):
    t, w = roots_jacobi(n, b - 1.0, a - 1.0)
    return L * (1 + t) / 2, w / w.sum()


LAWS = [
    Gam(2.0, r"Gamma, $V=2$", "#8c564b", "o"),
    Gam(1.0, r"Gamma, $V=1$  ($=\mathrm{Exp}(1)$)", "#ff7f0e", "s"),
    Gam(1.0 / 3, r"Gamma, $V=1/3$", "#d4b106", "^"),
    Nod(*beta_nodes(2.0, 6.0, 4.0), r"Beta$(2,6)$ with support on $(0,4)$",
        "#2ca02c", "D"),
    Nod(np.array([0.4, 2.4]), np.array([0.7, 0.3]),
        r"two-atom $\{0.4,\,2.4\}$", "#1f77b4", "P"),
    Nod(*beta_nodes(0.5, 0.3, 1.6), r"Beta$(0.5,0.3)$ with support on $(0,1.6)$",
        "#9467bd", "v"),
]


def solve(law, k, case):
    m, Dmu, Lph = law.data(k, case)
    f = lambda u: Dmu(u) + m * u / R0
    hi = 1.0
    while f(hi) < 0:
        hi *= 2
    u = brentq(f, 1e-12, hi, xtol=1e-15)
    return m, 1.0 - Lph(u)


Aplus = lambda m: 1.0 - np.exp(-TST / m)

# --------------------------------------------------------------- verification
print(f"R0 = {R0}   t* = {TST:.6f}   Z* = {ZSTAR:.6f}\n")
for law in LAWS:
    mm = 1.0 if isinstance(law, Gam) else float(law.pi @ law.x)
    print(f"  {law.name:<44} mean={mm:.10f}  b={law.b}")

KA = np.concatenate([[0.0], np.logspace(-2, np.log10(80), 400)])
KB = np.logspace(-2, np.log10(300), 400)
TRACE = {}
print("\nverification of A <= A^+ :")
worst = np.inf
for case, ks, tag in (("b", KA, "(a)  g = c x^k"), ("a", KB, "(b)  g = c e^{-kx}")):
    print(f"\n  panel {tag}")
    for law in LAWS:
        ms, As = zip(*[solve(law, k, case) for k in ks])
        ms, As = np.array(ms), np.array(As)
        TRACE[(law.name, case)] = (ms, As)
        gap = np.array([Aplus(m) for m in ms]) - As
        worst = min(worst, gap.min())
        print(f"    {law.name:<44} m in [{ms.min():.3f},{ms.max():8.3f}]  "
              f"min(A^+ - A) = {gap.min():+.3e}")
print(f"\n  smallest margin overall: {worst:+.3e}  "
      f"({'no violations' if worst > -1e-9 else 'VIOLATION'})")

kcross = brentq(lambda k: (1 + k) / (2 + k) - ZSTAR, 0, 50)
print(f"\n  panel (b): Exp(1) family has A = (1+k)/(2+k); "
      f"A > Z* for k > {kcross:.4f}")

# --------------------------------------------------------------------- figure
mpl.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
                     "legend.fontsize": 8.2, "figure.dpi": 130, "savefig.dpi": 130,
                     "mathtext.fontset": "dejavusans"})
fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.1))

PANELS = [
    dict(case="b", ax=axes[0], xscale="linear", xlim=(1.0, 5.4),
         shade="#2ca02c",
         title=r"(a)  $\mathcal{R}_0=2$,  $g(x)=c\,x^{k}$,  $k\geq 0$",
         marks=[]),
    dict(case="a", ax=axes[1], xscale="log", xlim=(0.04, 1.06),
         shade="#d32f2f",
         title=r"(b)  $\mathcal{R}_0=2$,  $g(x)=c\,e^{-kx}$,  $k>0$",
         marks=[]),
]

for P in PANELS:
    ax, case = P["ax"], P["case"]
    lo, hi = P["xlim"]
    mgrid = (np.linspace(lo, hi, 600) if P["xscale"] == "linear"
             else np.logspace(np.log10(lo), np.log10(hi), 600))
    AP = np.array([Aplus(m) for m in mgrid])

    ax.fill_between(mgrid, 0, AP, color=P["shade"], alpha=0.09, lw=0, zorder=0)
    ax.axhline(ZSTAR, color="0.45", ls="--", lw=1.3, zorder=2)
    ax.plot(mgrid, AP, color="#c62828", lw=2.6, zorder=6)

    for law in LAWS:
        ms, As = TRACE[(law.name, case)]
        o = np.argsort(ms)
        ms, As = ms[o], As[o]
        ax.plot(ms, As, color=law.color, lw=1.8, zorder=4)
        idx = [int(np.clip(np.searchsorted(ms, t), 0, len(ms) - 1))
               for t in P["marks"] if ms.min() <= t <= ms.max()]
        if idx:
            ax.plot(ms[idx], As[idx], color=law.color, marker=law.marker,
                    ls="none", ms=6, mec="white", mew=0.7, zorder=5)


    ax.set_xscale(P["xscale"])
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, 1)
    ax.set_title(P["title"])
    ax.set_xlabel(r"$m_1=\mathbb{E}(XY)=\mathbb{E}(X\,g(X))$"
                  + ("   (log scale)" if P["xscale"] == "log" else ""))
    ax.set_ylabel("attack ratio")
    ax.grid(alpha=0.13, lw=0.6)



style = [
    Line2D([], [], color="0.45", ls="--", lw=1.3,
           label=rf"homogeneous final size: $Z^{{*}}={ZSTAR:.3f}$"),
    Line2D([], [], color="#c62828", lw=2.6,
           label=rf"bound $A^{{+}}=1-e^{{-{TST:.3f}/m_1}}$"),
]
laws = [Line2D([], [], color=law.color, lw=1.8, marker=law.marker, ms=6,
               mec="white", mew=0.7, label=law.name) for law in LAWS]

axes[0].legend(handles=style + laws, loc="upper right", framealpha=0.95,
               handlelength=2.6, borderpad=0.6)
axes[1].legend(handles=style + laws, loc="lower left", framealpha=0.95,
               handlelength=2.6, borderpad=0.6)

plt.show()