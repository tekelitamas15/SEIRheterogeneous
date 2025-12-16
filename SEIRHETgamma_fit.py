import numpy as np
from scipy import optimize
from SEIRHETmodel import solve_SEIRhet
from SEIRHETplus import solve_SEIRhetplus
from SEIRHETplus2 import solve_SEIRhetplus

def seirhet_fit(data_dates, data_cases, N, initial_guess):
    """
    Fits the SEIR-HET model to the given data using curve fitting.

    Parameters:
        data_dates (array-like): Time points of the observed data.
        data_cases (array-like): Observed cases corresponding to the time points.
        N (int): Total population size.
        initial_guess (list): Initial guesses for the parameters [beta, gamma, alpha, p, E0].

    Returns:
        dict: Fitted parameters and their values.
    """

    # Create a wrapper for the solve_SEIRhet function
    def wrapped_model(x, beta, p, E0):
        return solve_SEIRhet(x, beta, 4, 4, p, E0, N, ur = 1)

    popt, _ = optimize.curve_fit(
        wrapped_model, data_dates, data_cases, p0=initial_guess,
        bounds=((3, 0, 0), (20, 10, N)), maxfev=5000
    )

    fitted_values = wrapped_model(data_dates, *popt)
    return fitted_values, popt

def seirhetplus_fit(data_dates, data_cases, N, initial_guess):
    """
    Fits the SEIR-HET model to the given data using curve fitting.

    Parameters:
        data_dates (array-like): Time points of the observed data.
        data_cases (array-like): Observed cases corresponding to the time points.
        N (int): Total population size.
        initial_guess (list): Initial guesses for the parameters [beta, gamma, alpha, p, E0].

    Returns:
        dict: Fitted parameters and their values.
    """

    # Create a wrapper for the solve_SEIRhet function
    def wrapped_model(x, beta, p, E0):
        return solve_SEIRhetplus(x, beta, 4, 4, p, E0, N, ur = 1)

    popt, _ = optimize.curve_fit(
        wrapped_model, data_dates, data_cases, p0=initial_guess,
        bounds=((3, 0, 0), (20, 10, N)), maxfev=5000
    )

    fitted_values = wrapped_model(data_dates, *popt)
    return fitted_values, popt