# This script finds the correlations between samples of three different species. 

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats
import os
import sys
import math

# Import the data mu_1.csv, mu_2.csv, and mu_3.csv and combine them into one data frame
data1 = pd.read_csv(r"C:\Users\User\Dropbox\UCL\GPs\TestingIndependentFitsGPR\Testoutput\outputs_GPR\simulations0\Y_1mu1.csv")
data2 = pd.read_csv(r"C:\Users\User\Dropbox\UCL\GPs\TestingIndependentFitsGPR\Testoutput\outputs_GPR\simulations0\Y_2mu1.csv")
data3 = pd.read_csv(r"C:\Users\User\Dropbox\UCL\GPs\TestingIndependentFitsGPR\Testoutput\outputs_GPR\simulations0\Y_3mu1.csv")
data = pd.concat([data1, data2, data3], axis=1)
data.columns = ["Y_1", "Y_2", "Y_3"]


# Find the correlations
correlations = data.corr()

# Print the correlations
print(correlations)

# Plot the correlations
sns.heatmap(correlations, xticklabels=correlations.columns, yticklabels=correlations.columns, cmap="YlGnBu")
plt.title("Correlations between species")
plt.show()


# plot using matshow from matplotlib
plt.matshow(correlations)
plt.xticks(range(len(correlations.columns)), correlations.columns)
plt.yticks(range(len(correlations.columns)), correlations.columns)
plt.colorbar()
plt.show()

# Use statistical methods to understand if there are correlations between the samples mu_1, mu_2, and mu_3
# The null hypothesis is that the samples are independent
# The alternative hypothesis is that the samples are not independent

# Find the p-values for the correlations
p_values = np.zeros((3,3))
for i in range(3):
    for j in range(3):
        p_values[i,j] = stats.pearsonr(data.iloc[:,i], data.iloc[:,j])[1]

# Print the p-values
print(p_values)

# Plot the p-values
sns.heatmap(p_values, xticklabels=correlations.columns, yticklabels=correlations.columns, cmap="YlGnBu")
plt.title("P-values for correlations between species")
plt.show()

def cross_correlation(series_a, series_b, lag):
    """
    Computes the cross-correlation between two series at a specified lag.
    """
    # Ensure that the series are numpy arrays
    series_a, series_b = np.asarray(series_a), np.asarray(series_b)
    
    # Remove means
    series_a, series_b = series_a - np.mean(series_a), series_b - np.mean(series_b)
    
    # Standardize by the standard deviations
    series_a, series_b = series_a / np.std(series_a), series_b / np.std(series_b)
    
    # Shift the series so that they align at the specified lag:
    if lag > 0:
        series_b = series_b[lag:]
        series_a = series_a[:-lag]
    elif lag < 0:
        series_b = series_b[:lag]
        series_a = series_a[-lag:]
    
    # Compute the cross-correlation
    correlation = np.mean(series_a * series_b)
    
    return correlation

def time_lagged_cross_correlation(series_a, series_b, max_lag):
    """
    Computes the time-lagged cross-correlation between two series up to a specified maximum lag.
    """
    correlations = [cross_correlation(series_a, series_b, lag) for lag in range(-max_lag, max_lag + 1)]
    return np.array(correlations), np.arange(-max_lag, max_lag + 1)

def pairwise_cross_correlation(series_dict, max_lag):
    results = {}
    series_names = list(series_dict.keys())
    for i in range(len(series_names)):
        for j in range(i + 1, len(series_names)):
            series_a_name, series_b_name = series_names[i], series_names[j]
            series_a, series_b = series_dict[series_a_name], series_dict[series_b_name]
            correlations, lags = time_lagged_cross_correlation(series_a, series_b, max_lag)
            results[(series_a_name, series_b_name)] = (correlations, lags)
    return results

# Example usage:
series_a = data1
series_b = data2
series_c = data3

series_dict = {'series_a': series_a, 'series_b': series_b, 'series_c': series_c}
max_lag = 5
correlation_results = pairwise_cross_correlation(series_dict, max_lag)

# Plotting the results
for pair, (correlations, lags) in correlation_results.items():
    plt.figure()
    plt.plot(lags, correlations)
    plt.title(f'Cross-correlation between {pair[0]} and {pair[1]}')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.show()

