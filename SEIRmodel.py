from scipy import integrate
import numpy as np
from scipy import optimize


def seir_model(y, x, beta, gamma, alpha, N):
    """
    Defines the SEIR model.

    Parameters:
        y (tuple): Current state [S, E, I, R].
        x (float): Time point.
        beta (float): Infection rate.
        gamma (float): Recovery rate.
        alpha (float): Rate of exposed individuals becoming infectious.
        N (int): Total population size.

    Returns:
        tuple: The derivatives of S, E, I, R.
    """
    S = (-1.0) * beta * y[0] * y[2] / N
    E = beta * y[0] * y[2] / N - alpha * y[1]
    I = alpha * y[1] - gamma * y[2]
    R = gamma * y[2]
    return S, E, I, R

def solve_SEIR(x, beta, gamma, alpha, E0, R0, N, ur):
    """
    Solves the SEIR model numerically over time.

    Parameters:
        x (array-like): Time points for integration.
        beta (float): Infection rate.
        gamma (float): Recovery rate.
        alpha (float): Rate of exposed individuals becoming infectious.
        E0 (float): Initial number of exposed individuals.
        N (int): Total population size.

    Returns:
        array: Fitted number of cases (Exposed + Infected) over time.
    """

    I0 = E0
    S0 = N - E0 - I0 - R0
    results = integrate.odeint(seir_model, (S0, E0, I0, R0), x, args=(beta, gamma, alpha, N))

    # diff = [1 - ur * S0]
    # for i in range(1, len(results[:, 0])):
    #     diff.append(results[:, 3][i])
    # return diff

    diff = [ur * E0]
    for i in range(1, len(results[:, 3])):
        diff.append(results[:, 3][i] - results[:, 3][i-1])
    return diff

    # diff = [ur * E0 + ur * I0]
    # for i in range(1, len(results[:, 2])):
    #     diff.append(results[:, 1][i] + results[:, 2][i])
    # return diff









