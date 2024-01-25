import numpy as np
import pymc as pm
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import pretty_errors


# Function to Load and Preprocess Data
def load_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    return data.values

# Function to Define Bayesian VAR Model


def define_bayesian_var_model(data, num_vars, priors_std):
    with pm.Model() as model:
        # Priors for VAR coefficients
        x0 = pm.Normal('x0', mu=0, sigma=1, shape=(num_vars, 1))
        A = pm.Normal('coeffs', mu=0, sigma=priors_std,
                      shape=(num_vars, num_vars))

        noise_chol, _, _ = pm.LKJCholeskyCov(
            "noise_chol", eta=1.0, n=num_vars, sd_dist=pm.HalfNormal.dist(sigma=1.0))

        mu = x0 + pm.math.dot(A, data[:-1, :].T)

        # Reshaping data for modeling
        # data_lagged = data.shift(1).dropna()
        # data_current = data.iloc[1:]

        # Likelihood of the observed data
        pm.MvNormal('likelihood', mu=mu.T,
                    chol=noise_chol, observed=data[1:, :])

    # Model fitting using MCMC sampling
    with model:
        trace = pm.sample(2000, tune=1000, cores=4)
    print(az.summary(trace, var_names=["x0", "coeffs"]))

    az.plot_posterior(trace, var_names=["x0", "coeffs"])
    plt.savefig("posterior_plot.pdf")
    return model

# Main Function


def main():
    # Load data
    data = load_data(
        r'C:\Users\User\Dropbox\UCL\GPs\synthetic_data\test_GMLV\simulations0.csv')

    # Define Bayesian VAR model
    num_vars = data.shape[1]
    model = define_bayesian_var_model(data, num_vars, priors_std=1)

    # # Results and visualization
    # # inference_data = az.from_pymc3(trace)
    # print(az.summary(trace, var_names=["x0", "A"]))
    # az.plot_posterior(trace, var_names=["x0", "A"])
    # plt.savefig("test_posterior_plot.pdf")
    # plt.show()

    # Further analysis and processing as required


if __name__ == '__main__':
    main()

    
