import numpy as np
from metrics import evaluate_fit  # Assume this calculates R-squared and other metrics
from SEIRmodel import solve_SEIR
from SEIRHETmodel import solve_SEIRhet
from SEIRHETplus2 import solve_SEIRhetplus
from SEIRHETgamma_fit import seirhet_fit, seirhetplus_fit
from SEIRHETuni_fit import seirhetuni_fit, seirhetuniplus_fit
from SEIRHETuni import solve_SEIRhetuni
from SEIRHETuniplus import solve_SEIRhetuniplus

from SEIR_fit import seir_fit
ur = 3.8

k = 2


def best_SEIR(data_dates, data_dates_test, data_cases, data_cases_test_v2, init_beta, N, E0_range, R0_range):
    best_r2_score = -np.inf
    best_E0 = None
    best_R0 = None
    best_beta = None
    best_fitted = None
    total_iterations = len(E0_range) * len(R0_range)  # Total number of iterations
    iteration = 0  # Counter

    for E0 in E0_range:
        for R0 in R0_range:  # Fixing the duplicate 'E0' loop
            iteration += 1  # Increment counter
            progress = (iteration / total_iterations) * 100  # Calculate progress



            fitted_values_seir, popt_seir = seir_fit(data_dates_test, data_cases_test_v2, N,
                                                     [init_beta, E0, R0])
            beta_seir, E0_seir, R0_seir = popt_seir
            r2_score = evaluate_fit(data_cases_test_v2,
                                    solve_SEIR(data_dates_test, beta_seir, 4, 4, E0_seir, R0_seir, N,
                                               ur=1))
            r2_score = r2_score['R-squared']
            print(f"Progress: {progress:.2f}% completed", end="\r")

            if r2_score is not None and r2_score > best_r2_score:
                best_r2_score = r2_score
                best_beta = beta_seir,
                best_E0 = E0_seir
                best_R0 = R0_seir
                best_fitted = fitted_values_seir
            print(f"SEIR-hom: {progress:.2f}% completed")  # Print progress in the same line

    return {
        "best_r2_score": best_r2_score,
        "best_beta": best_beta,
        "best_E0": best_E0,
        "best_R0": best_R0,
        "best_fitted": best_fitted
    }


def best_SEIRHET(data_dates, data_dates_test, data_cases, data_cases_test_v2, init_beta, N, E0_range, p_range):
    best_r2_score = -np.inf
    best_E0 = None
    best_p = None
    best_beta = None
    best_fitted = None
    total_iterations = len(E0_range) * len(p_range)  # Total iterations
    iteration = 0  # Counter

    for E0 in E0_range:
        for p in p_range:
            iteration += 1  # Increment counter
            progress = (iteration / total_iterations) * 100  # Compute progress percentage



            fitted_values_seirhet, popt_seirhet = seirhet_fit(data_dates_test, data_cases_test_v2, N,
                                                              [init_beta, p, E0])
            beta_seirhet, p_seirhet, E0_seirhet = popt_seirhet
            r2_score = evaluate_fit(data_cases_test_v2,
                                    solve_SEIRhet(data_dates_test, beta_seirhet, 4, 4, p_seirhet,
                                                  E0_seirhet, N, ur=1))
            r2_score = r2_score['R-squared']

            if r2_score is not None and r2_score > best_r2_score:
                best_r2_score = r2_score
                best_E0 = E0_seirhet
                best_p = p_seirhet
                best_beta = beta_seirhet
                best_fitted = fitted_values_seirhet
            print(f"SEIR-het (gamma, k = 1).: {progress:.2f}% completed")  # Print progress in the same line

    return {
        "best_r2_score": best_r2_score,
        "best_E0": best_E0,
        "best_p": best_p,
        "best_beta": best_beta,
        "best_fitted": best_fitted
    }

def best_SEIRHETPLUS(data_dates, data_dates_test, data_cases, data_cases_test_v2, init_beta, N, E0_range, p_range):
    best_r2_score = -np.inf
    best_E0 = None
    best_p = None
    best_beta = None
    best_fitted = None
    all_results = []

    total_iterations = len(E0_range) * len(p_range)  # Total iterations
    iteration = 0  # Counter

    for E0 in E0_range:
        for p in p_range:
            iteration += 1  # Increment counter
            progress = (iteration / total_iterations) * 100  # Compute progress percentage



            fitted_values_seirhet, popt_seirhet = seirhetplus_fit(data_dates_test, data_cases_test_v2, N,
                                                              [init_beta, p, E0])
            beta_seirhetplus, p_seirhetplus, E0_seirhetplus = popt_seirhet
            r2_score = evaluate_fit(data_cases_test_v2,
                                    solve_SEIRhetplus(data_dates_test, beta_seirhetplus, 4, 4, p_seirhetplus,
                                                  E0_seirhetplus, N, ur=1)).get('R-squared')
            all_results.append({
                "r2_score": r2_score,
                "beta": beta_seirhetplus,
                "p": p_seirhetplus,
                "E0": E0_seirhetplus,
                "fitted_values": fitted_values_seirhet
            })

            if r2_score is not None and r2_score > best_r2_score:
                best_r2_score = r2_score
                best_E0 = E0_seirhetplus
                best_p = p_seirhetplus
                best_beta = beta_seirhetplus
                best_fitted = fitted_values_seirhet

            print(f"SEIR-het (gamma, k = 2): {progress:.2f}% completed")  # Print progress in the same line

    return {
        "best_r2_score": best_r2_score,
        "best_E0": best_E0,
        "best_p": best_p,
        "best_beta": best_beta,
        "best_fitted": best_fitted
    }

def fit_SEIRHETUNI_b_L(data_dates_test, data_cases_test_v2, data_cases_v2, N, k, init_beta,
                        a_range, b_range, s0_range):

    best_r2_score = -np.inf
    best_a = None
    best_b = None
    best_beta = None
    best_s0 = None
    best_fitted = None
    total_iterations = len(a_range) * len(b_range) * len(s0_range)  # Total iterations
    iteration = 0  # Counter
    for a in a_range:
        for b in b_range:
            for s0 in s0_range:
                iteration += 1  # Increment counter
                progress = (iteration / total_iterations) * 100  # Compute progress percentage


                fitted_values, popt = seirhetuni_fit(
                    data_dates_test, data_cases_v2, N, k, [init_beta, a, b, s0]
                )
                beta_fit, a_fit, b_fit, s0_fit = popt
                r2_score = evaluate_fit(data_cases_v2, fitted_values).get('R-squared')

                if r2_score is not None and r2_score > best_r2_score:
                    best_r2_score = r2_score
                    best_a = a_fit
                    best_b = b_fit
                    best_beta = beta_fit
                    best_s0 = s0_fit
                    best_fitted = fitted_values

            print(f"SEIR-het (beta, k = {k}): {progress:.2f}% completed")  # Print progress in the same line
    return {
        "best_r2_score": best_r2_score,
        "best_a": best_a,
        "best_b": best_b,
        "best_beta": best_beta,
        "best_s0": best_s0,
        "fitted_values": best_fitted
    }

def fit_SEIRHETUNIPLUS(data_dates_test, data_cases_test_v2, data_cases_v2, N, k, init_beta,
                        a_range, b_range, s0_range):

    best_r2_score = -np.inf
    best_a = None
    best_b = None
    best_beta = None
    best_s0 = None
    best_fitted = None
    total_iterations = len(a_range) * len(b_range) * len(s0_range)  # Total iterations
    iteration = 0  # Counter
    for a in a_range:
        for b in b_range:
            for s0 in s0_range:
                iteration += 1  # Increment counter
                progress = (iteration / total_iterations) * 100  # Compute progress percentage


                fitted_values, popt = seirhetuniplus_fit(
                    data_dates_test, data_cases_v2, N, k, [init_beta, a, b, s0]
                )
                beta_fit, a_fit, b_fit, s0_fit = popt
                r2_score = evaluate_fit(data_cases_v2, fitted_values).get('R-squared')

                if r2_score is not None and r2_score > best_r2_score:
                    best_r2_score = r2_score
                    best_a = a_fit
                    best_b = b_fit
                    best_beta = beta_fit
                    best_s0 = s0_fit
                    best_fitted = fitted_values

            print(f"SEIR-het (beta, k = {k}): {progress:.2f}% completed")  # Print progress in the same line
    return {
        "best_r2_score": best_r2_score,
        "best_a": best_a,
        "best_b": best_b,
        "best_beta": best_beta,
        "best_s0": best_s0,
        "fitted_values": best_fitted
    }
