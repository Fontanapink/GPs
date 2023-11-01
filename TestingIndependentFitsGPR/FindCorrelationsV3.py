import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def time_lagged_cross_correlation(series_a, series_b, max_lag):
    series_a = np.asarray(series_a).squeeze()
    series_b = np.asarray(series_b).squeeze()
    lags = np.arange(-max_lag, max_lag + 1)
    correlations = [np.corrcoef(series_a, np.roll(series_b, lag))[0, 1] for lag in lags]
    return correlations, lags

def pairwise_cross_correlation_optimal_lag(series_dict, max_lag):
    results = {}
    optimal_lags = {}
    series_names = list(series_dict.keys())
    for i in range(len(series_names)):
        for j in range(len(series_names)):
            if i != j:
                series_a_name, series_b_name = series_names[i], series_names[j]
                series_a, series_b = series_dict[series_a_name], series_dict[series_b_name]
                correlations, lags = time_lagged_cross_correlation(series_a, series_b, max_lag)
                optimal_idx = np.argmax(np.abs(correlations))
                optimal_correlation = correlations[optimal_idx]
                optimal_lag_value = lags[optimal_idx]
                results[(series_a_name, series_b_name)] = (correlations, lags)
                optimal_lags[(series_a_name, series_b_name)] = (optimal_correlation, optimal_lag_value)
    return results, optimal_lags

def generate_correlation_matrix(optimal_lags, series_names):
    num_series = len(series_names)
    correlation_matrix = np.zeros((num_series, num_series))
    for i in range(num_series):
        for j in range(num_series):
            if i != j:
                pair = (series_names[i], series_names[j])
                optimal_corr, _ = optimal_lags[pair]
                correlation_matrix[i, j] = optimal_corr
            else:
                correlation_matrix[i, j] = 1
    return pd.DataFrame(correlation_matrix, columns=series_names, index=series_names)

# Load your data
data1 = pd.read_csv(r"C:\Users\User\Dropbox\UCL\GPs\TestingIndependentFitsGPR\Testoutput\outputs_GPR\simulations0\Y_1mu1.csv")
data2 = pd.read_csv(r"C:\Users\User\Dropbox\UCL\GPs\TestingIndependentFitsGPR\Testoutput\outputs_GPR\simulations0\Y_2mu1.csv")
data3 = pd.read_csv(r"C:\Users\User\Dropbox\UCL\GPs\TestingIndependentFitsGPR\Testoutput\outputs_GPR\simulations0\Y_3mu1.csv")

# Calculate the cross-correlations
series_dict = {'series_a': data1, 'series_b': data2, 'series_c': data3}
max_lag = 5
correlation_results, optimal_lags = pairwise_cross_correlation_optimal_lag(series_dict, max_lag)

# Generate and print the correlation matrix
correlation_matrix = generate_correlation_matrix(optimal_lags, list(series_dict.keys()))
print(correlation_matrix)

# Plot the results with optimal lags highlighted
for pair, (correlations, lags) in correlation_results.items():
    plt.figure(figsize=(10,5))
    plt.plot(lags, correlations, label='Cross-correlation')
    optimal_corr, optimal_lag = optimal_lags[pair]
    plt.scatter(optimal_lag, optimal_corr, color='red', label=f'Optimal Lag = {optimal_lag}')
    plt.title(f'Cross-correlation between {pair[0]} and {pair[1]}')
    plt.xlabel('Lag')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
