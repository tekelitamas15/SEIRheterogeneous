"""
Single panel.  Four admissible pairs (P_X, g) sharing V=1/3 and the first two
moments of mu, m_1=1.15 and m_2=1.62.  All four attack ratios lie in the
common band

    Z^-  =  [1 - exp(-t*(R0)(1+V) m_1/m_2)]/(1+V)     (lower bound)
    Z^+  =  1 - exp(-t*(R0)/m_1)                       (upper bound)

drawn against R0.
"""
import numpy as np
from scipy.optimize import brentq, linprog
from scipy.special import roots_jacobi
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

V = 1.0 / 3
M1, M2 = 1.15, 1.62


def tstar(R):
    return brentq(lambda t: -np.expm1(-t) - t / R, 0.5 * (R - 1), 12 * R, xtol=1e-15)


def beta_nodes(a, b, L, n=420):
    t, w = roots_jacobi(n, b - 1.0, a - 1.0)
    return L * (1 + t) / 2, w / w.sum()


def gamma_nodes(V, n=340):
    p = 1.0 / V
    j = np.arange(n)
    a = (2 * j + p) / p
    b = np.sqrt(j[1:] * (j[1:] + p - 1)) / p
    t, W = np.linalg.eigh(np.diag(a) + np.diag(b, 1) + np.diag(b, -1))
    return t, W[0] ** 2


class Law:
    def __init__(s, x, pi, name):
        s.x, s.pi, s.name = x, pi / pi.sum(), name
        s.V = float(s.pi @ x**2 - (s.pi @ x) ** 2)


LAWS = [
    Law(*gamma_nodes(V), r"Gamma, $V=1/3$"),
    Law(*beta_nodes(1.0, 1.0, 2.0), r"Uniform$(0,2)$"),
    Law(*beta_nodes(2.0, 6.0, 4.0), r"Beta$(2,6)$ on $(0,4)$"),
    Law(*beta_nodes(0.5, 0.3, 1.6), r"Beta$(0.5,0.3)$ on $(0,1.6)$"),
]
COL = ["#c9a227", "#e8710a", "#2e7d32", "#7b1fa2"]
MRK = ["^", "s", "D", "v"]


def normalise(pi, g):
    with np.errstate(divide="ignore"):
        lw = np.log(np.where(pi > 0, pi, 1e-300)) + np.log(np.where(g > 0, g, 1e-300))
    lw -= lw.max()
    q = np.exp(lw)
    return q / q.sum()


def match_g(law, nk=110):
    x, pi = law.x, law.pi
    ts = np.unique(np.round(np.quantile(x, np.linspace(0, 0.998, nk)), 8))
    B = np.column_stack([np.ones_like(x)]
                        + [(x > t).astype(float) for t in ts]
                        + [np.maximum(x - t, 0.0) for t in ts])
    Aeq = np.vstack([pi @ B, pi @ (x[:, None] * B), pi @ (x[:, None] ** 2 * B)])
    nt = len(ts)
    r = linprog(np.r_[0.0, 12.0 * np.ones(nt), np.ones(nt)],
                A_eq=Aeq, b_eq=[1.0, M1, M2],
                bounds=[(0, None)] * B.shape[1], method="highs")
    assert r.status == 0, law.name
    return B @ r.x


def attack(law, g, R0):
    q = normalise(law.pi, g)
    m = q @ law.x
    f = lambda u: float(q @ np.expm1(-u * law.x)) + m * u / R0
    hi = 1.0
    while f(hi) < 0:
        hi *= 2
    u = brentq(f, 1e-13, hi, xtol=1e-16)
    return 1 - float(law.pi @ np.exp(-u * law.x))


def Zplus(R0):
    return 1 - np.exp(-tstar(R0) / M1)


def Zminus(R0):
    return (1 - np.exp(-tstar(R0) * (1 + V) * M1 / M2)) / (1 + V)


# ---- compute
GS = {law.name: match_g(law) for law in LAWS}
R0s = np.linspace(1.03, 6.0, 400)
CURVE = [np.array([attack(law, GS[law.name], R) for R in R0s]) for law in LAWS]
ZP = np.array([Zplus(R) for R in R0s])
ZM = np.array([Zminus(R) for R in R0s])
ZKAT = np.array([(1 - np.exp(-tstar(R))) / (1 + V) for R in R0s])   # Z*/(1+V), g==1

print(f"V={V:.4f}  m1={M1}  m2={M2}  m2/m1={M2/M1:.5f}\n")
for law in LAWS:
    q = normalise(law.pi, GS[law.name])
    print(f"  {law.name:<30} Eg={law.pi@GS[law.name]:.6f}  "
          f"m1={q@law.x:.6f}  m2={q@law.x**2:.6f}")
for law, Zc in zip(LAWS, CURVE):
    print(f"  {law.name:<30} min(Z-Z^-)={np.min(Zc-ZM):+.2e}  min(Z^+-Z)={np.min(ZP-Zc):+.2e}")

# ---- figure
mpl.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12,
                     "legend.fontsize": 9.5, "figure.dpi": 130, "savefig.dpi": 130,
                     "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(8.4, 6.0))

ax.fill_between(R0s, ZM, ZP, color="#455a64", alpha=0.10, lw=0, zorder=0)
ax.plot(R0s, ZP, color="#37474f", lw=2.6, zorder=6)
ax.plot(R0s, ZM, color="#37474f", lw=2.6, ls=(0, (5, 2)), zorder=6)

for law, Zc, c, mk in zip(LAWS, CURVE, COL, MRK):
    ax.plot(R0s, Zc, color=c, lw=1.9, zorder=4)
    idx = [np.argmin(abs(R0s - t)) for t in (1.4, 1.9, 2.6, 3.5, 4.6, 5.6)]
    ax.plot(R0s[idx], Zc[idx], color=c, marker=mk, ls="none", ms=6.5,
            mec="white", mew=0.8, zorder=5)

ax.set_xlim(1.03, 6)
ax.set_ylim(0, 1)
ax.set_xlabel(r"$\mathcal{R}_0$")
ax.set_ylabel("attack ratio")
ax.set_title(r"Four distributions with shared $\mathrm{Var}(X)=1/3$, "
             r"$m_1=23/20$, $m_2=81/50$", pad=12)
ax.grid(alpha=0.14, lw=0.6)

handles = [
    Line2D([], [], color="#37474f", lw=2.6, label=r"upper bound $Z^{+}$"),
    Line2D([], [], color="#37474f", lw=2.6, ls=(0, (5, 2)), label=r"lower bound $Z^{-}$"),
] + [Line2D([], [], color=c, lw=1.9, marker=mk, ms=6.5, mec="white", mew=0.8,
            label=law.name) for law, c, mk in zip(LAWS, COL, MRK)]
ax.legend(handles=handles, loc="lower right", framealpha=0.96, handlelength=2.6)

plt.show()