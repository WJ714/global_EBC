import numpy as np
from sklearn.utils import resample
import warnings
warnings.simplefilter("ignore")
import xarray as xr
import pandas as pd
from itertools import product
import statsmodels.api as sm
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.interpolate import interp1d
from statsmodels.robust.robust_linear_model import RLM
from scipy.stats import f
from multiprocessing import Pool

def lowess_fit_with_filled_confidence_intervals(_x, _y, _weight, frac=4/5, num_bootstrap=50, confidence_level=0.95):
    # Fit LOWESS to the original data
    # weights_sqrt = np.sqrt(_weight)

    # # Apply weights to x and y
    # _x = _x * weights_sqrt
    # _y = _y * weights_sqrt
    lowess_results = lowess(_y, _x, return_sorted=True, it=50, delta=0.1, frac=frac)
    sorted_x = lowess_results[:, 0]#np.linspace(0.1,0.9, 1000)  # Cover the full range [0, 1]
    lowess_interp_func = interp1d(lowess_results[:, 0], lowess_results[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate",)
    fitted_y = lowess_interp_func(sorted_x)
    
    # Bootstrap to estimate confidence intervals
    bootstrap_estimates = np.zeros((num_bootstrap, len(sorted_x)))
    for i in range(num_bootstrap):
        # Resample data with replacement
        resampled_x, resampled_y = resample(_x, _y)
        # Fit LOWESS to resampled data
        resampled_lowess_results = lowess(resampled_y, resampled_x, return_sorted=True, it=50, delta=0.1, frac=frac)
        resampled_lowess_interp_func = interp1d(resampled_lowess_results[:, 0], resampled_lowess_results[:, 1], kind='linear', bounds_error=False, fill_value="extrapolate")
        bootstrap_estimates[i] = resampled_lowess_interp_func(sorted_x)
    
    # Calculate the percentiles for the confidence intervals
    lower_percentile = (1 - confidence_level) / 2 * 100
    upper_percentile = (1 + confidence_level) / 2 * 100
    lower_confidence = np.percentile(bootstrap_estimates, lower_percentile, axis=0)
    upper_confidence = np.percentile(bootstrap_estimates, upper_percentile, axis=0)
    
    # Use interp1d to interpolate and extrapolate the confidence intervals
    valid_x = sorted_x[~np.isnan(lower_confidence)]
    lower_conf_interp_func = interp1d(valid_x, lower_confidence[~np.isnan(lower_confidence)], kind='linear', bounds_error=False, fill_value="extrapolate")
    upper_conf_interp_func = interp1d(valid_x, upper_confidence[~np.isnan(upper_confidence)], kind='linear', bounds_error=False, fill_value="extrapolate")
    
    lower_confidence_filled = lower_conf_interp_func(sorted_x)
    upper_confidence_filled = upper_conf_interp_func(sorted_x)
    
    return lowess_interp_func, sorted_x, fitted_y, lower_confidence_filled, upper_confidence_filled

# Example usage
# _x, _y = your_data_x, your_data_y
# lowess_interp_func, sorted_x, fitted_y, lower_confidence_filled, upper_confidence_filled = lowess_fit_with_filled_confidence_intervals(_x, _y)


def robust_linear_regression(y, x):
    # Remove NaNs
    mask = ~np.isnan(y) & ~np.isnan(x)
    y_clean = y[mask]
    x_clean = x[mask]
    
    if len(y_clean) < 2 or len(x_clean) < 2:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    x_clean = sm.add_constant(x_clean)
    model = sm.RLM(y_clean, x_clean, M=sm.robust.norms.HuberT())
    results = model.fit()
    
    # Calculate Pearson correlation coefficient (r)
    correlation_matrix = np.corrcoef(y_clean, results.predict(x_clean))
    r = correlation_matrix[0, 1]
    
    # Perform an OLS regression to approximate the F-test for the R-squared
    ols_model = sm.OLS(y_clean, x_clean).fit()
    f_value = ols_model.fvalue
    f_pvalue = ols_model.f_pvalue
    
    return results.params[1], results.params[0], results.pvalues[1], results.pvalues[0], r, f_pvalue

# Function to apply linear regression across all time series
def linear_regression_all_time_series(gpcc_q, LE):
    slopes = np.full(LE.shape[1], np.nan)
    intercepts = np.full(LE.shape[1], np.nan)
    slope_pvalues = np.full(LE.shape[1], np.nan)
    intercept_pvalues = np.full(LE.shape[1], np.nan)
    r = np.full(LE.shape[1], np.nan)
    r_pvalues = np.full(LE.shape[1], np.nan)
    
    for b in range(LE.shape[1]):  # Loop over basin_id
        y = gpcc_q[:, b]
        x = LE[:, b]
        slope, intercept, slope_pval, intercept_pval, r2, r2_pval = robust_linear_regression(y, x)
        slopes[b] = slope
        intercepts[b] = intercept
        slope_pvalues[b] = slope_pval
        intercept_pvalues[b] = intercept_pval
        r[b] = r2
        r_pvalues[b] = r2_pval
                
    return slopes, intercepts, slope_pvalues, intercept_pvalues, r, r_pvalues

from scipy import stats
import pymannkendall as mk

# Define a function to compute Theil-Sen slope and p-value for a 1D array
def theil_sen_trend_with_significance(y, x):
    mask = np.isfinite(y)
    if np.sum(mask) < 2:
        return np.nan, np.nan
    slope, intercept, _, _ = stats.theilslopes(y[mask], x[mask])
    
    # Perform Mann-Kendall test for significance
    result = mk.original_test(y[mask])
    p_value = result.p

    return slope, p_value

            
def merge_small_bins_with_closest_neighbor(counts, bins, min_count):
    new_bins = [bins[0]]
    i = 0
    while i < len(counts):
        if counts[i] >= min_count:
            # Bin meets the minimum count, add the next boundary
            new_bins.append(bins[i+1])
            i += 1
        else:
            # Bin does not meet the minimum count, decide to merge
            if i == len(counts) - 1:
                # Last bin, simply adjust the last boundary to include this bin
                new_bins[-1] = bins[i+1]
                i += 1
            else:
                # Not the last bin, decide based on closest neighbor
                diff_prev = abs(counts[i-1] - counts[i]) if i > 0 else float('inf')
                diff_next = abs(counts[i+1] - counts[i])
                
                if diff_prev <= diff_next:
                    # Merge with the previous bin by not adding a new boundary
                    # No action needed here since we're merging with the previous bin
                    i += 1
                else:
                    # Merge with the next bin
                    # Skip adding the next boundary and move to the bin after next
                    i += 2
                    if i < len(counts):
                        new_bins.append(bins[i])

    # Ensure the last bin is always included
    if new_bins[-1] != bins[-1]:
        new_bins.append(bins[-1])
    return np.array(new_bins)

def budyko_et_p(ETp, P):
    # Compute the ratio ETp/P
    #refer to: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020WR028221
    ETp_P = ETp / P
    
    # Calculate the expression inside the square root
    exp_term = 1 - np.exp(-ETp_P)
    tanh_term = np.tanh(P / ETp)
    inner_expression = ETp_P * exp_term * tanh_term
    
    # Take the square root of the expression
    ET_P = np.sqrt(inner_expression)
    ET = ET_P*P
    return ET