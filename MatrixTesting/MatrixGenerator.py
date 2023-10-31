import numpy as np

def transform_to_matrix(filename):
    # Load the numpy object file
    data = np.load(filename)

    # Ensure the data is a list of lists
    if not all(isinstance(i, np.ndarray) for i in data):
        raise ValueError("The data is not a numpy array!")

    # Process each list in the data
    matrices = []
    for sublist in data:
        # Ensure the sublist has 15 parameters
        if len(sublist) != 15:
            raise ValueError(f"One of the sublists does not have 15 parameters: {sublist}")

        # Drop the first and last 3 values
        trimmed_list = sublist[3:-3]

        # Convert the 9 values into a 3x3 matrix
        matrix = np.matrix(np.reshape(trimmed_list, (3, 3)))

        # Set the diagonal elements to 1
        np.fill_diagonal(matrix, 1)
        
        matrices.append(matrix)
    
    # Save the matrices as a .npy file
    np.save("matrices.npy", matrices)
    return matrices

# Test the function with a sample .npy file
matrices = transform_to_matrix(r'C:\Users\User\Dropbox\UCL\GPs\MatrixTesting\parms.npy')
for matrix in matrices:
    print(matrix)
