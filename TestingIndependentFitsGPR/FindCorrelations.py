# This script finds the correlations between samples of three different species.

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import scipy.stats as stats


# Import the data mu_1.csv, mu_2.csv, and mu_3.csv and combine them into one data frame
data1 = pd.read_csv(
    r"C:\Users\User\Dropbox\UCL\GPs\TestingIndependentFitsGPR\Testoutput\outputs_GPR\simulations0\Y_1mu1.csv")
data2 = pd.read_csv(
    r"C:\Users\User\Dropbox\UCL\GPs\TestingIndependentFitsGPR\Testoutput\outputs_GPR\simulations0\Y_2mu1.csv")
data3 = pd.read_csv(
    r"C:\Users\User\Dropbox\UCL\GPs\TestingIndependentFitsGPR\Testoutput\outputs_GPR\simulations0\Y_3mu1.csv")
data = pd.concat([data1, data2, data3], axis=1)
data.columns = ["Y_1", "Y_2", "Y_3"]


# Find the correlations
correlations = data.corr()

# Print the correlations
print(correlations)

# Plot the correlations
sns.heatmap(correlations, xticklabels=correlations.columns,
            yticklabels=correlations.columns, cmap="YlGnBu")
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
p_values = np.zeros((3, 3))
for i in range(3):
    for j in range(3):
        p_values[i, j] = stats.pearsonr(data.iloc[:, i], data.iloc[:, j])[1]

# Print the p-values
print(p_values)

# Plot the p-values
sns.heatmap(p_values, xticklabels=correlations.columns,
            yticklabels=correlations.columns, cmap="YlGnBu")
plt.title("P-values for correlations between species")
plt.show()
