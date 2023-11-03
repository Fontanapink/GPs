import numpy as np
import scipy.stats as stats


def matrix_properties(matrix):
    # Ensure matrix has ones on the diagonal
    assert np.allclose(np.diag(matrix), 1), "Matrix does not have 1s on the diagonal"
    
    properties = {}
    #properties["matrix"] = matrix
    
    # Determinant
    properties["determinant"] = np.linalg.det(matrix)

    # Eigenvalues
    #properties["eigenvalues"] = np.linalg.eigvals(matrix)

    # Positive and negative values
    properties["positive_values_count"] = np.sum(matrix > 0)
    properties["negative_values_count"] = np.sum(matrix < 0)

    # Trace
    properties["trace"] = np.trace(matrix)

    # Rank
    properties["rank"] = np.linalg.matrix_rank(matrix)

    # Norms
    properties["l1_norm"] = np.linalg.norm(matrix, ord=1)
    properties["l2_norm"] = np.linalg.norm(matrix, ord=2)
    properties["frobenius_norm"] = np.linalg.norm(matrix, ord='fro')
    # Condition number
    properties["condition_number"] = np.linalg.cond(matrix)
    # Inverse
    # try:
    #     properties["inverse"] = np.linalg.inv(matrix)
    # except:
    #     properties["pseudo_inverse"] = np.linalg.pinv(matrix)
    # Singular values
    # properties["singular_values"] = np.linalg.svd(matrix, compute_uv=False)

    # Off-diagonal elements
    off_diagonal = matrix[np.where(~np.eye(matrix.shape[0],dtype=bool))]
    properties["off_diagonal_avg"] = np.mean(off_diagonal)
    properties["off_diagonal_variance"] = np.var(off_diagonal)

    # Sparsity
    properties['zero_elements'] = np.sum(matrix == 0)
    
    
    return properties

if __name__ == "__main__":
    matrices = np.load(r'C:\Users\User\Dropbox\UCL\GPs\MatrixTesting\matrices.npy', allow_pickle=True)
    # Save all matrices properties in a list
    matrix_properties_list = []
    for matrix in matrices:
        properties = matrix_properties(matrix)
        # Do something with the properties or print them
        
        matrix_properties_list.append(properties)
    # Save the list to a file
    np.save(r'C:\Users\User\Dropbox\UCL\GPs\MatrixTesting\matrix_properties_list.npy', matrix_properties_list)


    import pandas as pd

    # Convert data to DataFrame
    df = pd.DataFrame.from_dict(matrix_properties_list)

    # load other values from a csv file
    other_values_list = pd.read_csv(r'C:\Users\User\Dropbox\UCL\GPs\MatrixTesting\outputAutoregressive.csv')

    new_df = other_values_list.loc[other_values_list['Species'] == 'Species 1']
    new_df.reset_index(drop=True, inplace=True)
    df['kernel'] = new_df['Kernel']

    # # Calculate correlation
    # correlation = df.corr(method='pearson')  # Pearson is standard, but you can use 'spearman' or 'kendall' too

    # print(correlation['kernel'])

    results = {}
    # For each property, perform an ANOVA
    for col in df.columns:
        if col != 'kernel':
            groups = [df[col][df['kernel'] == group] for group in df['kernel'].unique()]
            f_stat, p_val = stats.f_oneway(*groups)
            results[col] = {'F-statistic': f_stat, 'p-value': p_val}

    # Print results
    for prop, vals in results.items():
        print(f"For property {prop}:")
        print(f"F-statistic = {vals['F-statistic']:.4f}")
        print(f"p-value = {vals['p-value']:.4f}\n")


    # Perform one-hot encoding on 'group'
    ohe = pd.get_dummies(df['kernel'], prefix='kernel')

    # Concatenate the OHE matrix with the original DataFrame
    df_ohe = pd.concat([df, ohe], axis=1).drop('kernel', axis=1)

    # Calculate correlations between matrix properties and OHE columns
    correlation_matrix = df_ohe.corr()

    # Extract correlations related to the 'group' OHE columns
    correlations_with_group = correlation_matrix[ohe.columns].loc[df.columns.difference(['kernel'])]

    print(correlations_with_group)