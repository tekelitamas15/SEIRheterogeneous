import numpy as np
from scipy import optimize
from SEIRHETuni import solve_SEIRhetuni
from SEIRHETuniplus import solve_SEIRhetuniplus

def seirhetuni_fit(data_dates, data_cases, N, k, initial_guess):
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
    def wrapped_model(x, beta, a, b, s0):
        result = solve_SEIRhetuni(x, beta, 4, 4, k, a, b, N, s0, 1)
        return np.nan_to_num(result, nan=0, posinf=0, neginf=0)

    popt, _ = optimize.curve_fit(
        wrapped_model, data_dates, data_cases, p0=initial_guess,
        bounds=((1e-2, 1e-2, 1e-2, 1e-7), (8, 6, 5, 1))
    )

    fitted_values = wrapped_model(data_dates, *popt)

    return fitted_values, popt

def seirhetuniplus_fit(data_dates, data_cases, N, k, initial_guess):
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
    def wrapped_model(x, beta, a, b, s0):
        result = solve_SEIRhetuniplus(x, beta, 4, 4, k, a, b, N, s0, 1)
        return np.nan_to_num(result, nan=0, posinf=0, neginf=0)

    popt, _ = optimize.curve_fit(
        wrapped_model, data_dates, data_cases, p0=initial_guess,
        bounds=((1e-2, 1e-2, 1e-2, 1e-7), (8, 5, 5, 1))
    )

    fitted_values = wrapped_model(data_dates, *popt)

    return fitted_values, popt
