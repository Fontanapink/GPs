import numpy as np

def matrix_properties(matrix):
    # Ensure matrix has ones on the diagonal
    assert np.allclose(np.diag(matrix), 1), "Matrix does not have 1s on the diagonal"
    
    properties = {}
    properties["matrix"] = matrix
    
    # Determinant
    properties["determinant"] = np.linalg.det(matrix)

    # Eigenvalues
    properties["eigenvalues"] = np.linalg.eigvals(matrix)
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
    try:
        properties["inverse"] = np.linalg.inv(matrix)
    except:
        properties["pseudo_inverse"] = np.linalg.pinv(matrix)
    # Singular values
    properties["singular_values"] = np.linalg.svd(matrix, compute_uv=False)

    # Off-diagonal elements
    off_diagonal = matrix[np.where(~np.eye(matrix.shape[0],dtype=bool))]
    properties["off_diagonal_avg"] = np.mean(off_diagonal)
    properties["off_diagonal_variance"] = np.var(off_diagonal)

    # Sparsity
    properties['zero_elements'] = np.sum(matrix == 0)
    
    return properties

if __name__ == "__main__":
    matrices = np.load(r'C:\Users\User\Dropbox\UCL\GPs\MatrixTesting\matrices.npy', allow_pickle=True)
    for matrix in matrices:
        properties = matrix_properties(matrix)
        # Do something with the properties or print them
        print(properties)
