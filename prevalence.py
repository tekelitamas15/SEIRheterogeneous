"""
Governing equations (closed in scalars sbar,U,V):
    sbar' = -beta sbar V
    U'    =  beta Phi_1(sbar) V - alpha U
    V'    =  alpha U - gamma V
with  Phi_k(w) = int x^k g(x) w^x phi_X(x) dx,  U=int g e phi, V=int g i phi.

Prevalence recovery (closed aggregates, driven by sbar,V):
    E' = beta V Psi_1(sbar) - alpha E      Psi_1(w)=int x w^x phi dx   (no g)
    I' = alpha E - gamma I
    S(t) = Phi_0-type without g = int sbar^x phi dx ;  R = 1-S-E-I.
Plotted: V(t) (first and third row), I(t) (second and fourth row).

Couplings (E g(X)=1):  g=1, g=x, g=c e^{-2x}.  beta = R0 gamma / m_1, m_1=Phi_1(1)=E(XY).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import roots_genlaguerre, gammaln
import matplotlib as mpl
import matplotlib.pyplot as plt

R0, alpha, gamma = 1.5, 4.0, 4.0
E0 = 0.000002
T = 25


def quad(V, n=200):
    """Gauss-Laguerre nodes/weights for X~Gamma(p,p), p=1/V, plus homogeneous V=0."""
    if V == 0:
        return np.array([1.0]), np.array([1.0])
    p = 1.0 / V
    t, w = roots_genlaguerre(n, p - 1.0)
    x = t / p
    pi = w / np.exp(gammaln(p))
    return x, pi / pi.sum()


def coupling(x, kind):
    if kind == "one":
        return np.ones_like(x)
    if kind == "lin":
        return x.copy()
    return np.exp(-2.0 * x)


def run(V, kind):
    x, pi = quad(V)
    g = coupling(x, kind)
    g = g / (pi @ g)                       # E g(X)=1
    m = pi @ (x * g)                       # = Phi_1(1) = E(XY)
    beta = R0 * gamma / m

    # scalar integrals as functions of w=sbar:
    def Phi1(w):                           # int x g w^x phi
        return pi @ (x * g * w**x)
    def Psi1(w):                           # int x w^x phi   (no g)
        return pi @ (x * w**x)
    def Sfun(w):                           # int w^x phi  = S(t)
        return pi @ (w**x)


    sbar0 = 1.0
    U0 = E0 * (pi @ g) / 2
    Vv0 = E0 * (pi @ g) / 2
    Eagg0 = E0 / 2
    Iagg0 = E0 / 2
    y0 = [sbar0, U0, Vv0, Eagg0, Iagg0]

    def rhs(t, y):
        sbar, U, Vv, Ea, Ia = y
        sbar = max(sbar, 1e-12)
        dsbar = -beta * sbar * Vv
        dU = beta * Phi1(sbar) * Vv - alpha * U
        dV = alpha * U - gamma * Vv
        dEa = beta * Vv * Psi1(sbar) - alpha * Ea
        dIa = alpha * Ea - gamma * Ia
        return [dsbar, dU, dV, dEa, dIa]

    sol = solve_ivp(rhs, [0, T], y0, rtol=1e-10, atol=1e-13,
                    dense_output=True, max_step=0.02)
    ts = np.linspace(0, T, 800)
    P = np.empty_like(ts); R = np.empty_like(ts); Aux = np.empty_like(ts)
    for j, t in enumerate(ts):
        sbar, U, Vv, Ea, Ia = sol.sol(t)
        Aux[j] = Vv
        P[j] = Ea
        R[j] = 1.0 - Sfun(max(sbar, 1e-12)) - Ea - Ia
    return ts, Aux, P, R, beta, m

tdist = 1.5

print(f"early growth rate d/dt log(E+I) at t~{tdist}, per coupling:")
for kind, lab in [("one", "g=1"), ("lin", "g=x"), ("exp", "g=e^-2x")]:
    row = f"  {lab:<8}: "
    for V in [0, 1/3, 1, 3]:
        ts, Aux, P, R, beta, m = run(V, kind)
        r = (np.log(np.interp(tdist + 0.2, ts, P)) - np.log(np.interp(tdist - 0.2, ts, P))) / 0.4
        Auxr = (np.log(np.interp(tdist + 0.2, ts, Aux)) - np.log(np.interp(tdist - 0.2, ts, Aux))) / 0.4
        row += f"V= {V}: {r:.3f}  "
    print(row)

print(f"early growth rate d/dt log(U+V) at t~{tdist}, per coupling:")
for kind, lab in [("one", "g=1"), ("lin", "g=x"), ("exp", "g=e^-2x")]:
    row = f"  {lab:<8}: "
    for V in [0, 1/3, 1, 3]:
        ts, Aux, P, R, beta, m = run(V, kind)
        Auxr = (np.log(np.interp(0.7, ts, Aux)) - np.log(np.interp(0.3, ts, Aux))) / 0.4
        row += f"V={V}:{Auxr:.3f}  "
    print(row)


CASES = [(0, r"hom. SEIR", "#111111"), (1/3, r"$V=1/3$", "#d4b106"),
         (1, r"$V=1$", "#ff7f0e"), (3, r"$V=3$", "#1f77b4")]
COLS = [("one", r"$g(x)=1$"), ("lin", r"$g(x)=x$"), ("exp", r"$g(x)=c\,e^{-2x}$")]

mpl.rcParams.update({"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 11,
                     "legend.fontsize": 8.2, "figure.dpi": 130, "savefig.dpi": 130,
                     "mathtext.fontset": "dejavusans"})
fig, axes = plt.subplots(4, 3, figsize=(15.0, 10.0), sharex=True)


for col, (kind, ctitle) in enumerate(COLS):
    for V, lab, c in CASES:
        ts, Aux, P, R, beta, m = run(V, kind)
        axes[0, col].plot(ts, Aux, color=c, lw=2.1, label=lab)
        axes[1, col].plot(ts, P, color=c, lw=2.1, label=lab)
        axes[2, col].plot(ts, Aux, color=c, lw=2.1, label=lab)
        axes[2, col].set_yscale('log')
        axes[3, col].plot(ts, P, color=c, lw=2.1, label=lab)
        axes[3, col].set_yscale('log')
    axes[0, col].set_title(ctitle)

for col in range(3):
    axes[2, col].set_ylim(0, 0.18)
    axes[3, col].set_ylim(0, 0.85)
    axes[0, col].set_ylim(0, 0.04)
    axes[1, col].set_ylim(0, 0.04)
    axes[1, col].set_xlim(0, T)
    axes[3, col].set_xlabel(r"time (weeks)")
axes[0, 0].set_ylabel(r"proportion of pop.")
axes[1, 0].set_ylabel(r"proportion of pop.")
axes[2, 0].set_ylabel(r"proportion of pop.")
axes[3, 0].set_ylabel(r"proportion of pop.")
axes[0, 2].text(1.02, 0.5, r"$V(t)$", transform=axes[0, 2].transAxes,
                rotation=270, va="center", ha="left", fontsize=14.5)
axes[1, 2].text(1.02, 0.5, r"$I(t)$", transform=axes[1, 2].transAxes,
                rotation=270, va="center", ha="left", fontsize=14.5)
axes[2, 2].text(1.02, 0.5, r"$V(t)$", transform=axes[2, 2].transAxes,
                rotation=270, va="center", ha="left", fontsize=14.5)
axes[3, 2].text(1.02, 0.5, r"$I(t)$", transform=axes[3, 2].transAxes,
                rotation=270, va="center", ha="left", fontsize=14.5)

for ax in axes.flat:
    ax.grid(alpha=0.14, lw=0.6)
axes[0, 0].legend(loc="upper right", framealpha=0.95)
axes[1, 0].legend(loc="upper right", framealpha=0.95)
axes[2, 0].legend(loc="lower center", framealpha=0.95)
axes[3, 0].legend(loc="lower center", framealpha=0.95)
fig.suptitle(r"$\mathcal{R}_0=1.5$: independent infectiousness ($g=1$), "
             r"positively correlated ($g=x$), negatively correlated ($g=c\,e^{-2x}$)",
             y=0.995)
fig.tight_layout(rect=(0, 0, 0.97, 0.98))
fig.savefig("prev6b.png", bbox_inches="tight", dpi=100)





plt.show()