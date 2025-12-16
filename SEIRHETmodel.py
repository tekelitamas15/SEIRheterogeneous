from scipy import integrate
import numpy as np
from scipy import optimize


def seirhet_model(y, x, beta, gamma, alpha, p, N):
    """
    SEIR-HET model function that calculates the derivatives for a heterogeneous SEIR model.

    Parameters:
        y (list or ndarray): Current state of the compartments [S, E, I, R].
        x (float): Time (not used explicitly in this model but required for compatibility with odeint).
        beta (float): Transmission rate parameter.
        gamma (float): Recovery rate parameter.
        alpha (float): Incubation rate parameter.
        p (float): Heterogeneity parameter that modifies the transmission dynamics.
        N (int): Total population size.

    Returns:
        tuple: Derivatives of the compartments (dS/dt, dE/dt, dI/dt, dR/dt).
    """
    N = sum(y)  # Ensure the total population size remains consistent.
    S = -beta * pow((y[0]/N), 1 + (1 / p)) * y[2]
    E = beta * pow((y[0]/N), 1 + (1 / p)) * y[2] - alpha * y[1]
    I = alpha * y[1] - gamma * y[2]
    R = gamma * y[2]
    return S, E, I, R


def solve_SEIRhet(x, beta, gamma, alpha, p, E0, N, ur = 3.8):
    """
    Solves the SEIR-HET model using numerical integration.

    Parameters:
        x (array-like): Time points at which to solve the model.
        beta (float): Transmission rate parameter.
        gamma (float): Recovery rate parameter.
        alpha (float): Incubation rate parameter.
        p (float): Heterogeneity parameter that modifies the transmission dynamics.
        E0 (float): Initial number of exposed individuals.
        N (int): Total population size.

    Returns:
        ndarray: Sum of the Exposed (E) and Infectious (I) compartments at each time point.
    """
    R0 = 0
    I0 = E0
    S0 = N - E0 - I0 - R0

    # Solve the ODE system
    results = integrate.odeint(seirhet_model, (S0, E0, I0, R0), x, args=(beta, gamma, alpha, p, N))

    # diff = [1 - ur * S0]
    # for i in range(1, len(results[:, 3])):
    #     diff.append(results[:, 3][i])
    # return diff

    # diff = [ur * E0 + ur * I0]
    # for i in range(1, len(results[:, 0])):
    #     diff.append(results[:, 1][i] + results[:, 2][i])
    # return diff

    diff = [ur * E0]
    for i in range(1, len(results[:, 0])):
        diff.append(results[:, 3][i] - results[:, 3][i - 1])


    return diff
