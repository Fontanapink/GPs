
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
matrix_properties = np.load(
    r'C:\Users\User\Dropbox\UCL\GPs\MatrixTesting\matrix_properties_list.npy', allow_pickle=True)
matrices = np.load(
    r'C:\Users\User\Dropbox\UCL\GPs\MatrixTesting\matrices.npy', allow_pickle=True)
kernel_data = pd.read_csv(
    r'C:\Users\User\Dropbox\UCL\GPs\MatrixTesting\outputAutoregressive.csv')

# Map matrix properties to the kernel data
id_to_properties_map = {int(id_str.replace('simulations', '')): props for id_str, props in zip(
    kernel_data['Input File'].unique(), matrix_properties)}
kernel_data['Matrix Properties'] = kernel_data['Input File'].str.replace(
    'simulations', '').astype(int).map(id_to_properties_map)

# Expand the matrix properties into separate columns
properties_df = kernel_data['Matrix Properties'].apply(pd.Series)
kernel_data_expanded = kernel_data.join(properties_df)
kernel_data_expanded.drop('Matrix Properties', axis=1, inplace=True)

# ANOVA analysis with check for constant values
anova_results = {}
# Skip the first six columns which are not matrix properties
for col in kernel_data_expanded.columns[6:]:
    groups = [kernel_data_expanded[col][kernel_data_expanded['Kernel'] ==
                                        kernel].values for kernel in kernel_data_expanded['Kernel'].unique()]
    # Check for constant values within groups
    if any(np.std(group) == 0 for group in groups):
        continue  # Skip this property because at least one group is constant
    f_stat, p_val = stats.f_oneway(*groups)
    anova_results[col] = {'F-statistic': f_stat, 'p-value': p_val}


# One-hot encoding and correlation matrix calculation
ohe_kernel = pd.get_dummies(kernel_data_expanded['Kernel'], prefix='Kernel')
df_ohe = pd.concat([kernel_data_expanded.drop(
    ['Input File', 'Species', 'BIC', 'Kernel', 'Mean'], axis=1), ohe_kernel], axis=1)
correlation_matrix = df_ohe.corr()
correlations_with_kernel = correlation_matrix[ohe_kernel.columns].loc[df_ohe.columns.difference(
    ohe_kernel.columns)]

# Boxplot generation for matrix properties
properties_to_plot = [col for col in kernel_data_expanded.columns[6:]
                      if col not in ['trace', 'rank', 'zero_elements']]
n_rows = len(properties_to_plot) // 2 + len(properties_to_plot) % 2
n_cols = 2
fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(15, 5 * n_rows))
axes = axes.flatten()
for i, property_to_plot in enumerate(properties_to_plot):
    sns.boxplot(x='Kernel', y=property_to_plot,
                data=kernel_data_expanded, ax=axes[i])
    axes[i].set_title(
        f'Distribution of {property_to_plot} for each Kernel Type')
    axes[i].set_xticklabels(axes[i].get_xticklabels(), rotation=45)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
# save the plot
plt.savefig(
    r'C:\Users\User\Dropbox\UCL\GPs\MatrixTesting\matrix_properties_boxplots.png')
plt.show()
