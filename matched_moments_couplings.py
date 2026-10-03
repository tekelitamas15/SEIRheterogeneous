"""Non-decreasing couplings g(x) realising E g=1, m1=23/20, m2=81/50 with each distribution."""
import numpy as np
from scipy.optimize import linprog
from scipy.special import roots_jacobi
import matplotlib as mpl
import matplotlib.pyplot as plt

V = 1.0 / 3
M1, M2 = 1.15, 1.62


def beta_nodes(a, b, L, n=700):
    t, w = roots_jacobi(n, b - 1.0, a - 1.0)
    return L * (1 + t) / 2, w / w.sum()


def gamma_nodes(V, n=340):
    p = 1.0 / V
    j = np.arange(n)
    a = (2 * j + p) / p
    b = np.sqrt(j[1:] * (j[1:] + p - 1)) / p
    t, W = np.linalg.eigh(np.diag(a) + np.diag(b, 1) + np.diag(b, -1))
    return t, W[0] ** 2


LAWS = [
    ("Gamma, $V=1/3$", gamma_nodes(V), 6.0, "#c9a227"),
    ("Uniform$(0,2)$", beta_nodes(1.0, 1.0, 2.0), 2.0, "#e8710a"),
    ("Beta$(2,6)$ on $(0,4)$", beta_nodes(2.0, 6.0, 4.0), 4.0, "#2e7d32"),
    ("Beta$(0.5,0.3)$ on $(0,1.6)$", beta_nodes(0.5, 0.3, 1.6), 1.6, "#7b1fa2"),
]


def match_g_inc(x, pi, nk=110):
    """non-decreasing g >= 0 : basis of upper-tail indicators and hinges."""
    ts = np.unique(np.round(np.quantile(x, np.linspace(0, 0.998, nk)), 8))
    B = np.column_stack([np.ones_like(x)]
                        + [(x > t).astype(float) for t in ts]
                        + [np.maximum(x - t, 0.0) for t in ts])
    nt = len(ts)
    Aeq = np.vstack([pi @ B, pi @ (x[:, None] * B), pi @ (x[:, None] ** 2 * B)])
    r = linprog(np.r_[0.0, 12.0 * np.ones(nt), np.ones(nt)],
                A_eq=Aeq, b_eq=[1.0, M1, M2],
                bounds=[(0, None)] * B.shape[1], method="highs")
    assert r.status == 0
    return B @ r.x


mpl.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12.5,
                     "legend.fontsize": 10, "figure.dpi": 130, "savefig.dpi": 130,
                     "mathtext.fontset": "dejavusans"})
fig, ax = plt.subplots(figsize=(8.4, 5.6))

ax.axhline(1.0, color="0.55", ls=":", lw=1.1, zorder=1)
ax.annotate(r"independent infectiousness", xy=(2.6, 1.0), xytext=(0, 5),
            textcoords="offset points", color="0.4", fontsize=10.5)

for name, (x, pi), xmax, col in LAWS:
    g = match_g_inc(x, pi)
    o = np.argsort(x)
    m = x[o] <= xmax + 1e-9
    ax.plot(x[o][m], g[o][m], color=col, lw=2.2, zorder=4, label=name)
    ax.plot([x[o][m][-1]], [g[o][m][-1]], color=col, marker="o", ms=6,
            mec="white", mew=0.8, zorder=5)

ax.set_xlim(0, 4)
ax.set_ylim(0, 1.75)
ax.set_xlabel(r"susceptibility $x$")
ax.set_ylabel(r"$g(x)=\mathbb{E}(Y\mid X=x)$")
ax.set_title(r"$g(x)$ functions satisfying $\mathbb{E}(g(X))=1$, "
             r"$m_1=23/20$, $m_2=81/50$")
ax.grid(alpha=0.14, lw=0.6)
ax.legend(loc="lower right", framealpha=0.96)
plt.savefig("matched_moments_couplings.pdf")
plt.show()