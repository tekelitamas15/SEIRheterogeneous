from scipy import integrate
import numpy as np
from scipy import optimize
import math
from scipy.special import hyp1f1
from scipy.special import beta as scipy_beta



def Beta(a,b):
    return scipy_beta(a, b)

def phi_s(k, s, a, b):
    return ((1 + b/a) ** k) / Beta(a, b) * integrate.quad(lambda x: math.exp((1 + b/a) * np.log(s) * x) * (x ** (k + a -1)) * (1 - x) ** (b - 1), 0, 1)[0]


def my_hyp1f1_function(k, s, a, b):
    x = (1 + (b / a)) * np.log(s)
    if k == 0:
        return hyp1f1(a, a + b, x)
    elif k == 1:
        return hyp1f1(a + 1, a + b + 1, x)
    elif k == 2:
        multiplier = ( ((a+b) * (a+1)) / ((a+ b + 1) * a) )
        return multiplier * hyp1f1(a + 2, a + b + 2, x)
    else:
        raise ValueError("k must be either 1 or 2")

def S_from_s(s, a, b):
    return 1 / (Beta(a, b)) * integrate.quad(lambda x: math.exp( (1 + b/a) * np.log(s) * x) * x ** (a-1) * (1 - x) ** (b - 1), 0, 1)[0]

# Main ODE system: dy/dx = f(y, x)
def seirhetuni_model(y, t, beta, gamma, alpha, k, a, b):
    s, E, I, R = y

    dsdt = -beta * s * I
    dEdt = beta * my_hyp1f1_function(k, s, a, b) * I - alpha * E
    dIdt = alpha * E - gamma * I
    dRdt = gamma * I
    return [dsdt, dEdt, dIdt, dRdt]

def solve_SEIRhetuni(x, beta, gamma, alpha, k, a, b, N, s0, ur = 1):


    R0 = 0
    E0 = I0 = 0.5 * (1 - s0 )

    results = integrate.odeint(seirhetuni_model, (s0, E0, I0, R0), x, args=(beta, gamma, alpha, k, a, b))


    # diff = [ur * E0 + ur * I0]
    # for i in range(1, len(results[:, 2])):
    #     diff.append(results[:, 1][i] + results[:, 2][i])
    # return diff

    # output = [1 - S_from_s(s0, a, b)]
    # # diff = [R0]
    # for i in range(1, len(results[:, 0])):
    #     output.append(results[:, 3][i])
    #     # output.append(1 - S_from_s(results[:, 0][i], a, b))

    # return output

    diff = [ur * 0.5 * (1 - my_hyp1f1_function(0, s0, a, b)) * N]
    for i in range(1, len(results[:, 3])):
        diff.append(N * results[:, 3][i] - N * results[:, 3][i - 1])


    return diff




