from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def evaluate_fit(data_cases, fitted_values):
    """
    Evaluates the fit of the model using statistical metrics.

    Parameters:
        data_cases (array-like): Observed cases corresponding to the time points.
        fitted_values (array-like): Predicted values from the model.

    Returns:
        dict: R-squared error metrics.
    """
    r2 = r2_score(data_cases, fitted_values)

    return {
        "R-squared": r2
    }