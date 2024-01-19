import os
import numpy as np
import pymc as pm
import pytensor.tensor as at
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az

from pytensor.printing import Print


def generate_var1_data(n_obs, coefficients, initial_values, noise_stddev=1):
    """
    Generate simulated data from a VAR(1) process.

    Parameters:
    - n_obs (int): Number of observations to generate.
    - coefficients (numpy.ndarray): Coefficient matrix for the VAR(1) process.
                                    It should be a square matrix of shape (dim, dim).
    - initial_values (numpy.ndarray): Initial values for the process, a vector of shape (dim,).
    - noise_stddev (float): Standard deviation of the normally distributed noise term.

    Returns:
    - data (numpy.ndarray): Simulated data matrix of shape (n_obs, dim).
    """
    dim = len(initial_values)
    data = np.zeros((n_obs, dim))
    data[0, :] = initial_values[:, 0]

    for t in range(1, n_obs):
        # VAR(1) process: X_t = A * X_{t-1} + noise
        noise = np.random.normal(scale=noise_stddev, size=dim)
        data[t, :] = np.dot(coefficients, data[t - 1, :]) + noise

    return data


def simple_sim():
    """
    Simulates data using a VAR model.

    Example usage:
    np.random.seed(42)  # for reproducibility
    n_obs = 100
    coefficients = np.array([[0.8, -0.2], [0.3, 0.5]])
    initial_values = np.array([[1], [2]])

    simulated_data = generate_var1_data(n_obs, coefficients, initial_values)

    print("Simulated Data:")
    print(simulated_data)
    """
    np.random.seed(42)  # for reproducibility
    n_obs = 100
    coefficients = np.array([[0.8, -0.2], [0.3, 0.5]])
    initial_values = np.array([[1], [2]])

    simulated_data = generate_var1_data(n_obs, coefficients, initial_values)

    print("Simulated Data:")
    print(simulated_data)


def run_inference():
    """
    Runs the inference process for the VAR model.

    Returns:
    None
    """

    np.random.seed(42)
    n_obs = 100
    coefficients_true = np.array([[0.8, -0.2], [0.3, 0.5]])
    initial_values_true = np.array([1, 2]).reshape((2, 1))
    noise_stddev = 0.1

    data = generate_var1_data(n_obs, coefficients_true,
                              initial_values_true, noise_stddev)

    # PyMC3 model
    dim = len(initial_values_true)

    with pm.Model() as var_model:
        # Priors for x0 and sigma
        x0 = pm.Normal('x0', mu=0, sigma=1, shape=(dim, 1))
        A = pm.Normal('A', mu=0, sigma=1, shape=(dim, dim))

        # Priors for coefficients with LKJ prior
        # packed_L = pm.LKJCholeskyCov('packed_L', n=dim, eta=2.0, sd_dist=pm.HalfCauchy.dist(2.5))
        # L = pm.expand_packed_triangular(dim, packed_L)
        # coefficients = pm.MvNormal('coefficients', mu=0, chol=L, shape=(dim, dim))

        noise_chol, _, _ = pm.LKJCholeskyCov(
            "noise_chol", eta=1.0, n=dim, sd_dist=pm.HalfNormal.dist(sigma=1.0))

        # VAR(1) process likelihood
        print("x0:", x0.shape)
        # print("A:",A.shape)
        # print("data[:-1, :]:", data[:-1, :].shape)
        print("data[1:, :]:", data[1:, :].shape)
        x0_obs = data[0, :].copy().reshape(2, 1)
        print("x0:", x0_obs.shape)

        mu = x0 + pm.math.dot(A, data[:-1, :].T)
        print("mu:", mu.T.shape)
        print("data:", data[1:, :].shape)

        # obs_chol = np.diag(np.full(dim,sigma))

        # *pm.Normal('likelihood_0', mu=x0, sigma=1.0, observed=x0_obs)
        likelihood = pm.MvNormal(
            'likelihood_t', mu=mu.T, chol=noise_chol, observed=data[1:, :])

    # Sampling from the posterior
    with var_model:
        trace = pm.sample(2000, tune=1000, cores=2)

    # Plotting the posterior distributions
    # pm.plot_posterior(trace, var_names=['x0', 'A'])

    # Extracting the posterior means for initial values and coefficients
    # posterior_means = {
    #    'x0': np.mean(trace['x0'], axis=0),
    #    'A': np.mean(trace['A'], axis=0)
    #

    # print("Posterior Means:")
    # print(posterior_means)

    print(az.summary(trace, var_names=["x0", "A"]))

    az.plot_posterior(trace, var_names=["x0", "A"], ref_val=[
                      1, 2, 0.8, -0.2, 0.3, 0.5])
    plt.savefig("posterior_plot.pdf")


def run_inference_large():
    """
    Run large-scale inference for VAR model.

    This function generates VAR model data, fits a Bayesian VAR model using PyMC3,
    and performs posterior sampling and analysis.

    Returns:
        None
    """
    # np.random.seed(42)
    ndim = 5
    n_obs = 24
    coefficients_true = np.zeros((ndim, ndim))
    # np.fill_diagonal(coefficients_true, 1.0)
    coefficients_true[2, 3] = 0.8
    coefficients_true[1, 4] = -0.5

    print(coefficients_true)

    initial_values_true = np.ones((ndim, 1))*1.0
    noise_stddev = 0.1

    data = generate_var1_data(n_obs, coefficients_true,
                              initial_values_true, noise_stddev)

    fig, axs = plt.subplots(ndim, 1, figsize=(10, 2*ndim))
    for i, ax in enumerate(axs):
        axs[i].plot(data[:, i])
    plt.savefig("plot-data.pdf")

    # Params for shrinkage
    D = ndim*ndim
    D0 = 2
    N = n_obs

    # create and fit PyMC model
    with pm.Model() as var_model:
        # Standard LKJ priors Priors for x0 and sigma
        # x0 = pm.Normal('x0', mu=initial_values_true, sigma=0.01, shape=(ndim,1))
        # A = pm.Normal('A', mu=0, sigma=1, shape=(ndim, ndim))

        # Priors for coefficients with LKJ prior
        # noise_chol, _, _ = pm.LKJCholeskyCov("noise_chol", eta=1.0, n=dim, sd_dist=pm.HalfNormal.dist(sigma=1.0) )

        # VAR(1) process likelihood
        # mu = x0 + pm.math.dot(A, data[:-1, :].T)
        # likelihood = pm.MvNormal('likelihood_t', mu=mu.T, chol=noise_chol, observed=data[1:, :])

        # Priors for coefficients with horseshoe -> sparse VAR
        x0 = pm.Normal('x0', mu=initial_values_true,
                       sigma=0.001, shape=(ndim, 1))

        # Standard horse shoe
        # Prior on error SD
        # sigma = pm.HalfNormal("sigma", 25)
        # Global shrinkage prior
        # tau = pm.HalfStudentT("tau", 2, D0 / (D - D0) * sigma / np.sqrt(N))
        # Local shrinkage prior
        # lam = pm.HalfStudentT("lam", 5, shape=(ndim, ndim) )
        # c2 = pm.InverseGamma("c2", 2, 8)
        # z = pm.Normal("z", 0.0, 1.0, shape=(ndim, ndim) )
        # Shrunken coefficients
        # A = pm.Normal('A', mu=0, sigma = z * tau * lam * at.sqrt(c2 / (c2 + tau**2 * lam**2)), shape=(ndim, ndim) )

        # Regularised horse shoe
        tau0 = (D0 / (D - D0)) * noise_stddev / np.sqrt(N)
        c2 = pm.InverseGamma("c2", 2, 8)
        tau = pm.HalfCauchy("tau", beta=tau0)
        lam = pm.HalfCauchy("lam", beta=1, shape=(ndim, ndim))
        A = pm.Normal('A', mu=0, sigma=tau * lam*at.sqrt(c2 /
                      (c2 + tau**2 * lam**2)), shape=(ndim, ndim))

        # Priors for coefficients with LKJ prior
        noise_chol, _, _ = pm.LKJCholeskyCov(
            "noise_chol", eta=1.0, n=ndim, sd_dist=pm.HalfNormal.dist(sigma=1.0))

        # VAR(1) process likelihood
        mu = x0 + pm.math.dot(A, data[:-1, :].T)
        likelihood = pm.MvNormal(
            'likelihood_t', mu=mu.T, chol=noise_chol, observed=data[1:, :])

    # Sampling from the posterior
    with var_model:
        trace = pm.sample(2000, tune=1000, cores=4)

    print(az.summary(trace, var_names=["A"]))

    az.plot_posterior(trace, var_names=[
                      "A"], ref_val=coefficients_true.flatten().tolist())
    plt.savefig("plot-posterior.pdf")


if __name__ == '__main__':
    # simple_sim()
    run_inference()
    # run_inference_large()
