#
#
#   Temporal Series Utils
#
#

from math import sqrt

import numpy as np
from statsmodels.tsa.stattools import acf, pacf


def mad_mean_error(col_real, col_pred):
    """
    Calculate for each user, mean absolute deviation/mean ratio
    Is an alternative to MAPE, avoiding problems with values close to zero

    Args:
        col_real: (float list) with real consumption
        col_pred: (float list) with predicted consumption

    Returns:
        mad_mean_error: (float)

    """
    mad_mean = 0
    if col_real and col_pred and len(col_real) > 0 and len(col_pred) > 0:

        length = len(col_real)
        diff = [abs(col_real[i] - col_pred[i]) for i in range(length)]

        try:
            mad = sum(diff) / length

        except ZeroDivisionError:
            mad = 0

        try:
            mad_mean = mad / (sum(col_real) / length)  # +1 added

        except ZeroDivisionError:
            mad_mean = 0

    return float(mad_mean)


def nrmsd_error_norm(col_real, col_pred):
    """
    Calculate for each user,  root mean square deviation.
    Represents the sample standard deviation of the differences between predicted values
    and observed values. RMSD is sensitive to outliers. RMSD isn't an average error.
    Lower values will indicate less residual variance

    Args:
        col_real: (float list) with real consumption
        col_pred: (float list) with predicted consumption

    Returns:
        nrmsd_error_norm: (float)

    """
    nrmsd = 0
    if col_real and col_pred and len(col_real) > 0 and len(col_pred) > 0:
        length = len(col_real)
        sum_diff = 0
        sum_real = 0

        for i in range(length):
            diff = abs(col_real[i] - col_pred[i])
            diff2 = diff * diff
            sum_diff = sum_diff + diff2
            sum_real = col_real[i] + sum_real

        try:
            nrmsd = (sqrt(sum_diff / length)) / (max(col_real) - min(col_real))  # +1 added

        except ZeroDivisionError:
            nrmsd = 0

    return float(nrmsd)


def rmse_train(col_real, col_pred):
    """
    Calculate for each user,  quadratic mean error for execution date.

    Args:
        col_real: (float) with real consumption
        col_pred: (float) with predicted consumption

    Returns:
        rmse_value: (float) RMSE

    """
    rmse_value = 0
    if col_real and col_pred and len(col_real) > 0 and len(col_pred) > 0:
        length = len(col_real)
        diffs = [(i - j) * (i - j) for i in col_real for j in col_pred]
        sum_diffs = sum(diffs)
        rmse_value = sqrt(sum_diffs / length)

    return float(rmse_value)


def residual_analysis(col_real, col_pred):
    """
    This function checks normality and correlation in residuals.
    The model will be check:
        - Ljun-Box test
            H0: The data are independently distributed (i.e. correlations is close to 0)
            H1: The data are not independently distributed; they exhibit serial correlation
        - residuals mean is zero

    Args:
        col_real: (float list) with real consumption
        col_pred: (float list) with predicted consumption

    Returns:
        bool: True if reject H0. The model can improve
              False if we can't reject H0 and residuals can be independents

    """

    if col_real and col_pred and len(col_real) > 0 and len(col_pred) > 0:
        length = len(col_real)
        array_residuals = [col_real[i] - col_pred[i] for i in range(length)]

        array_p_values = acf(array_residuals, qstat=True)[2]
        len_value_with_corr = len([i for i in list(array_p_values) if i < 0.05])

        if array_residuals == [0] * len(array_residuals):
            return False

        # If mean error is close to zero and
        # if there are 5% or minus of p_values < statistical significance
        elif (int(np.mean(array_residuals)) == 0) & (len_value_with_corr <= 0.05 * len(array_residuals)):
            # We can reject null hypothesis (independence on residuals)
            return False
        else:
            # We can't reject null hypothesis (independence on residuals. We can't accept dependence.
            return True

    return True


def estimated_window(series, limit_inf_days):
    """
    Calculate the number of days in every window of Holt Winters, using partial autocorrelation
    of the series, i.e, parameter _slen_ of the model.

    Args:
        series: (float list) with real consumption
        limit_inf_days: (int) minimum size of the window. It is recommendable at least 2 (days)

    Returns:
        slen: (float) temporary window optimized for each user

    """

    try:
        auto_corr = pacf(series, int(len(series) / 2), method='ols')
        auto_corr = [float(abs(val)) for val in auto_corr]

    except np.linalg.linalg.LinAlgError as err:
        if 'Singular matrix' or 'SVD did not converge' in err.message:
            return -1
        else:
            raise Exception(str(err) + ' def' + str(series))

    if len(auto_corr) > (limit_inf_days - 1):
        m = max(auto_corr[(limit_inf_days - 1):int(len(auto_corr))])
        list_results = [i + 1 for i, j in enumerate(auto_corr) if j == m]
        if list_results[0] == len(auto_corr):
            return -1  # suppose that `series` is non stationary and it must not apply hw
        else:
            slen = list_results[0]
    else:
        return -1

    return int(slen)


def initial_trend(series, slen):
    """
    Calculate the initial trend of a series
    Args:
        series: (float list) containing the series of observed values (total_data_kb_qt) for one user
        slen: (int) size temporal window

    Returns:
        initial_trend: (float) initial trend for this series

    """

    trend = 0.0

    for i in range(slen):
        trend += float(series[i + slen] - series[i]) / slen

    ini_trend = trend / slen

    return ini_trend


def initial_seasonal_components(series, slen):
    """
    Initialize the seasonal components of the series

    Args:
        series: (float list) containing the series of observed values (total_data_kb_qt) for one user
        slen: (int) size temporal window

    Returns:
        seasonal_comp: (float list) with initial value with each point of seasonal component for one user

    """

    seasonal_comp = {}
    season_averages = []
    n_seasons = int(len(series) / slen)

    # Compute season averages
    for j in range(n_seasons):
        season_averages.append(sum(series[slen * j:slen * j + slen]) / float(slen))

    # Compute initial values
    for i in range(slen):

        sum_of_vals_over_avg = 0.0

        for j in range(n_seasons):
            sum_of_vals_over_avg += series[slen * j + i] - season_averages[j]

        seasonal_comp[i] = sum_of_vals_over_avg / n_seasons

    return seasonal_comp


# Here starts code licensed under the MIT License (../LICENSE.txt)
def rmse_holt_winters(params, *args):
    """
    This function returns the root square error of the model Holt Winters which should be minimized
    input parameter params: Initial values of alpha, beta and gamma before optimization
    input parameter *args: arguments of the function below fmin_l_bfgs_b which minimizes RMSE function
    """
    Y = args[0]
    types = args[1]
    y = []

    if types == 'linear':
        alpha, beta = params
        a = [Y[0]]
        b = [Y[1] - Y[0]]
        y = [a[0] + b[0]]
        for i in range(len(Y)):
            a.append(alpha * Y[i] + (1 - alpha) * (a[i] + b[i]))
            b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
            y.append(a[i + 1] + b[i + 1])
    else:
        alpha, beta, gamma = params
        m = args[2]
        a = [sum(Y[0:m]) / float(m)]
        b = [(sum(Y[m:2 * m]) - sum(Y[0:m])) / m ** 2]
        if types == 'additive':
            s = [Y[i] - a[0] for i in range(m)]
            y = [a[0] + b[0] + s[0]]
            for i in range(len(Y)):
                a.append(alpha * (Y[i] - s[i]) + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s.append(gamma * (Y[i] - a[i] - b[i]) + (1 - gamma) * s[i])
                y.append(a[i + 1] + b[i + 1] + s[i + 1])
        elif types == 'multiplicative':
            s = [Y[i] / a[0] for i in range(m)]
            y = [(a[0] + b[0]) * s[0]]
            for i in range(len(Y)):
                a.append(alpha * (Y[i] / s[i]) + (1 - alpha) * (a[i] + b[i]))
                b.append(beta * (a[i + 1] - a[i]) + (1 - beta) * b[i])
                s.append(gamma * (Y[i] / (a[i] + b[i])) + (1 - gamma) * s[i])
                y.append((a[i + 1] + b[i + 1]) * s[i + 1])
        else:
            pass
    rmse = sqrt(sum([(m - n) ** 2 for m, n in zip(Y, y[:-1])]) / len(Y))
    return rmse
# Here is the end of code licensed under the MIT License (../LICENSE.txt)
