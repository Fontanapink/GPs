# %%
import numpy as np
import math
from numpy.linalg import inv
import matplotlib.pyplot as plt
from numpy.linalg import cholesky, det
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from scipy.integrate import solve_ivp

import pretty_errors

import gpflow as gpf
from gpflow.utilities import print_summary
from gpflow.utilities import parameter_dict
from gpflow.ci_utils import reduce_in_tests

import tensorflow as tf

gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("notebook")
np.random.seed(0)

MAXITER = reduce_in_tests(5000)


# %%
# The idea is that we simulate from a lotka volterra model with three species. This model has three growth rates, mu, plus an interaction matrix, M
# We then fit Gaussian processes to the time courses and use model selection to determine the best combination of kernels and mean functions to model the data
# What we ultimately want to know is how the original parameters of the LV model correspond to the best fitting GPs
# This will enable us to work out what information is contained in the GPs

# %%
# This function is the Lotka-Volterra predator-prey model
# It takes two arguments: t, the time, and y, a vector of the current population sizes
# It returns a list of the time derivatives of the populations, in the same order as the input

def lotka_volterra(t, y):
    mu = [0.2, 0.7, 0.9]
    M = np.array([[-0.1, 0.0, 0.0], [0.0, -0.1, 0.1], [0.1, 0.0, -0.1]])

    y1 = y[0]
    y2 = y[1]
    y3 = y[2]

    dy1 = y1*mu[0] + y1*(M[0, 0]*y1 + M[0, 1]*y2 + M[0, 2]*y3)
    dy2 = y2*mu[1] + y2*(M[1, 0]*y1 + M[1, 1]*y2 + M[1, 2]*y3)
    dy3 = y3*mu[2] + y3*(M[2, 0]*y1 + M[2, 1]*y2 + M[2, 2]*y3)

    return [dy1, dy2, dy3]


def simulate(y0, t):
    return solve_ivp(fun=lotka_volterra, t_span=[min(t), max(t)], y0=y0, t_eval=t, method='LSODA')


nps = 31
t = np.linspace(0, 25, nps)
y0 = [10.0, 10.0, 10.0]
sol = simulate(y0, t)

# sample data points
# s_idx = np.random.choice(len(t), size = 101, replace=False)
# s_idx.sort()
s_idx = np.arange(nps)
ts = sol.t[s_idx]
ys = sol.y[:, s_idx]

# add noise to growth data
y_hat = np.maximum(ys + np.random.normal(scale=0.01, size=ys.shape), 0)

print(y_hat.shape)

fig, ax = plt.subplots(figsize=(15, 5), ncols=3, nrows=1)
ax[0].plot(ts, y_hat[0, :], "bx", mew=2)
ax[1].plot(ts, y_hat[1, :], "gx", mew=2)
ax[2].plot(ts, y_hat[2, :], "rx", mew=2)


# %%
# Fit whole system using various multi-ouput kernels and VGP


def plot_gp_d(x, mu, var, color, label, ax):
    ax.plot(x, mu, color=color, lw=2, label=label)
    ax.fill_between(
        x[:, 0],
        (mu - 2 * np.sqrt(var))[:, 0],
        (mu + 2 * np.sqrt(var))[:, 0],
        color=color,
        alpha=0.4,
    )
    ax.set_xlabel("X")
    ax.set_ylabel("y")


def plot_model(m, X, P, L, K_L, M_F, BIC):
    fig, ax = plt.subplots(figsize=(15, 5), ncols=3, nrows=1)
    ax[0].plot(X[:, 0], Y[:, 0], "bx", mew=2)
    ax[1].plot(X[:, 0], Y[:, 1], "gx", mew=2)
    ax[2].plot(X[:, 0], Y[:, 2], "rx", mew=2)

    # just use the GP to predict at same timepoints
    mu1, var1 = m.predict_y(np.hstack((X, np.zeros_like(X))))
    plot_gp_d(X, mu1, var1, "b", "Y1", ax[0])

    mu2, var2 = m.predict_y(np.hstack((X, np.ones_like(X))))
    plot_gp_d(X, mu2, var2, "g", "Y2", ax[1])

    mu3, var3 = m.predict_y(np.hstack((X, 2*np.ones_like(X))))
    plot_gp_d(X, mu3, var3, "r", "Y3", ax[2])

    fig.suptitle('species= ' + str(P) + ', latent_processes= ' + str(L) + ', kernel= ' +
                 str(K_L.__name__) + ', mean= ' + str(M_F.__class__.__name__) + ', BIC =' + str(BIC))


def optimize_model_with_scipy(model):
    optimizer = gpf.optimizers.Scipy()
    res = optimizer.minimize(
        model.training_loss_closure((X, Y)),
        variables=model.trainable_variables,
        method="l-bfgs-b",
        # options={"disp": 50, "maxiter": MAXITER},
        options={"maxiter": MAXITER},
    )
    return res


def count_params(m):
    p_dict = parameter_dict(m)
    p_count = 0
    for val in p_dict.values():
        # print(val.shape)
        if len(val.shape) == 0:
            p_count = p_count + 1
        else:
            p_count = p_count + math.prod(val.shape)

    return p_count

# This is for model selection: the lower the BIC the better the model


def get_BIC(m, F, n):
    # Assumes F = -lnL
    # QUESTION: is this correct? Are we sure it is model parameters and not number of kernels parameters?
    k = count_params(m)
    return k*np.log(n) + 2*F


# %%
# Here do coregionalization to estimate f(x) = W g(x)
# https://gpflow.github.io/GPflow/2.8.0/notebooks/advanced/multioutput.html
# https://gpflow.github.io/GPflow/develop/notebooks/getting_started/mean_functions.html
# https://towardsdatascience.com/sparse-and-variational-gaussian-process-what-to-do-when-data-is-large-2d3959f430e7
# https://gpflow.readthedocs.io/en/v1.5.1-docs/notebooks/advanced/coregionalisation.html
# https://gpflow.github.io/GPflow/2.4.0/notebooks/advanced/coregionalisation.html
# This uses VGP

X = ts.reshape(-1, 1)
Y = y_hat.T

print(X.shape)
print(Y.shape)

# Augment the input with ones or zeros to indicate the required output dimension
X_aug = np.vstack(
    (np.hstack((X, np.zeros_like(X))),
     np.hstack((X, np.ones_like(X))),
     np.hstack((X, 2*np.ones_like(X)))
     )
)

# Augment the Y data with ones or zeros that specify a likelihood from the list of likelihoods
Y1 = Y[:, 0].reshape(-1, 1)
Y2 = Y[:, 1].reshape(-1, 1)
Y3 = Y[:, 2].reshape(-1, 1)

Y_aug = np.vstack(
    (np.hstack((Y1, np.zeros_like(Y1))),
     np.hstack((Y2, np.ones_like(Y2))),
     np.hstack((Y3, 2*np.ones_like(Y3)))
     )
)

# print(X)
# print(X_aug)
# print(Y_aug)

# %%
L = 1  # latent processes, g in R^L
P = 3  # observed processes, f in R^P

# Base kernel
k = gpf.kernels.Matern32(active_dims=[0])
# k = gpf.kernels.SquaredExponential(active_dims=[0])

# Coregion kernel
coreg = gpf.kernels.Coregion(
    output_dim=P,
    rank=L,
    active_dims=[1]
)

kern = k * coreg

# This likelihood switches between Gaussian noise with different variances for each f_i:
lik = gpf.likelihoods.SwitchedLikelihood(
    [gpf.likelihoods.Gaussian(), gpf.likelihoods.Gaussian(),
     gpf.likelihoods.Gaussian()]
)

# now build the GP model as normal
m = gpf.models.VGP((X_aug, Y_aug), kernel=kern, likelihood=lik)

# fit the covariance function parameters
maxiter = reduce_in_tests(10000)
res = gpf.optimizers.Scipy().minimize(
    m.training_loss,
    m.trainable_variables,
    options=dict(maxiter=maxiter),
    method="L-BFGS-B",
)


print_summary(m)
# plot_model(m)
BIC = get_BIC(m, res.fun, X.shape[0])
print(BIC)
plot_model(m, X, P, L, gpf.kernels.Matern32, M_F=None, BIC=BIC)

# %%
# Wrap above code into a funtion
# P is the number of outputs (three in this case for the three species)
# L is the number of latent processes
# K_L is the kernel for the latent processes
# M_F is the mean function applied to latent processes


def fit_model(X_aug, Y_aug, P, L, K_L=gpf.kernels.SquaredExponential, M_F=None):

    # Base kernel for leatent processes
    # k = gpf.kernels.Matern32(active_dims=[0])
    # k = gpf.kernels.SquaredExponential(active_dims=[0])

    k = K_L(active_dims=[0])

    # Coregion kernel
    coreg = gpf.kernels.Coregion(
        output_dim=P,
        rank=L,
        active_dims=[1]
    )

    kern = k * coreg

    # This likelihood switches between Gaussian noise with different variances for each f_i:
    lik = gpf.likelihoods.SwitchedLikelihood(
        [gpf.likelihoods.Gaussian() for _ in range(P)]
    )

    # now build the GP model as normal
    m = gpf.models.VGP((X_aug, Y_aug), kernel=kern,
                       likelihood=lik, mean_function=M_F)

    # fit the covariance function parameters
    maxiter = reduce_in_tests(10000)
    res = gpf.optimizers.Scipy().minimize(
        m.training_loss,
        m.trainable_variables,
        options=dict(maxiter=maxiter),
        method="L-BFGS-B",
    )

    print_summary(m)
    BIC = get_BIC(m, res.fun, X.shape[0])
    print("BIC:", BIC)
    plot_model(m, X, P, L, K_L, M_F, BIC)

    return m, BIC


# Try different kernels
# k = gpf.kernels.Matern32(active_dims=[0])
# k = gpf.kernels.SquaredExponential(active_dims=[0])
# k = gpf.kernels.RationalQuadratic(active_dims=[0])
# k = gpf.kernels.Exponential(active_dims=[0])
# k = gpf.kernels.Linear(active_dims=[0])
# k = gpf.kernels.Cosine(active_dims=[0])
# k = gpf.kernels.Periodic(active_dims=[0])
# k = gpf.kernels.Polynomial(active_dims=[0])
# k = gpf.kernels.Matern12(active_dims=[0])
# k = gpf.kernels.Matern52(active_dims=[0])
# k = gpf.kernels.Brownian(active_dims=[0])
# k = gpf.kernels.White(active_dims=[0])
# k = gpf.kernels.Constant(active_dims=[0])
# k = gpf.kernels.Coregion(active_dims=[0])
# k = gpf.kernels.ChangePoints(active_dims=[0])
# k = gpf.kernels.LinearCoregionalization(active_dims=[0])


# %%
fit_model(X_aug, Y_aug, 3, 1, gpf.kernels.Matern32)

# %%
fit_model(X_aug, Y_aug, 3, 1, gpf.kernels.SquaredExponential,
          M_F=gpf.functions.Polynomial(2))

# %%
fit_model(X_aug, Y_aug, 3, 2, gpf.kernels.SquaredExponential,
          M_F=gpf.functions.Polynomial(2))

# %%
fit_model(X_aug, Y_aug, 3, 3, gpf.kernels.SquaredExponential,
          M_F=gpf.functions.Polynomial(2))

# %%
# Try different numbers of latent processes, L, with the same kernel, K_L = matern32 and mean function, M_F = polynomial
fit_model(X_aug, Y_aug, 3, 1, gpf.kernels.Matern32,
          M_F=gpf.functions.Polynomial(2))
fit_model(X_aug, Y_aug, 3, 2, gpf.kernels.Matern32,
          M_F=gpf.functions.Polynomial(2))
fit_model(X_aug, Y_aug, 3, 3, gpf.kernels.Matern32,
          M_F=gpf.functions.Polynomial(2))

# Try different numbers of latent processes, L, with the same kernel, K_L = squared exponential and mean function, M_F = polynomial
fit_model(X_aug, Y_aug, 3, 1, gpf.kernels.SquaredExponential,
          M_F=gpf.functions.Polynomial(2))
fit_model(X_aug, Y_aug, 3, 2, gpf.kernels.SquaredExponential,
          M_F=gpf.functions.Polynomial(2))
fit_model(X_aug, Y_aug, 3, 3, gpf.kernels.SquaredExponential,
          M_F=gpf.functions.Polynomial(2))


# %%
# Try different mean functions, kernels, and number of latent processes and see which one fits the data best using BIC as a metric for model selection (lower is better)
best_BIC = 100000000
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.SquaredExponential, M_F=gpf.functions.Polynomial(2))
if BIC < best_BIC:
    best_model = m
    best_BIC = BIC
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=2, K_L=gpf.kernels.SquaredExponential, M_F=gpf.functions.Polynomial(2))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=3, K_L=gpf.kernels.SquaredExponential, M_F=gpf.functions.Polynomial(2))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=4, K_L=gpf.kernels.SquaredExponential, M_F=gpf.functions.Polynomial(2))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=5, K_L=gpf.kernels.SquaredExponential, M_F=gpf.functions.Polynomial(2))

# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Matern32, M_F=gpf.functions.Polynomial(2))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=2, K_L=gpf.kernels.Matern32, M_F=gpf.functions.Polynomial(2))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=3, K_L=gpf.kernels.Matern32, M_F=gpf.functions.Polynomial(2))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=4, K_L=gpf.kernels.Matern32, M_F=gpf.functions.Polynomial(2))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=5, K_L=gpf.kernels.Matern32, M_F=gpf.functions.Polynomial(2))

# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.RationalQuadratic, M_F=gpf.functions.Polynomial(2))

# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.SquaredExponential, M_F=gpf.functions.Constant(0.5))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.SquaredExponential, M_F=gpf.functions.Constant(1))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.SquaredExponential, M_F=gpf.functions.Constant(1.5))

# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Matern32, M_F=gpf.functions.Constant(0.5))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Matern32, M_F=gpf.functions.Constant(1))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Matern32, M_F=gpf.functions.Constant(1.5))

# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.RationalQuadratic, M_F=gpf.functions.Constant(0.5))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.RationalQuadratic, M_F=gpf.functions.Constant(1))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.RationalQuadratic, M_F=gpf.functions.Constant(1.5))

# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.SquaredExponential, M_F=gpf.functions.Linear(0.5))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.SquaredExponential, M_F=gpf.functions.Linear(1))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.SquaredExponential, M_F=gpf.functions.Linear(1.5))

# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Matern32, M_F=gpf.functions.Linear(0.5))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Matern32, M_F=gpf.functions.Linear(1))
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Matern32, M_F=gpf.functions.Linear(1.5))


# %%
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.SquaredExponential, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Matern32, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.RationalQuadratic, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Exponential, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Linear, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Cosine, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Periodic, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Polynomial, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Matern12, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Matern52, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.Brownian, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gpf.kernels.White, M_F=None)

# %%
# Identifying the best kernel, mean function and latent process dimensionality for the data set using BIC score as the metric for comparison of models

best_BIC = 10000000
kernels = [gpf.kernels.SquaredExponential, gpf.kernels.Matern32, gpf.kernels.RationalQuadratic, gpf.kernels.Exponential, gpf.kernels.Linear,
           gpf.kernels.Cosine, gpf.kernels.Periodic, gpf.kernels.Polynomial, gpf.kernels.Matern12, gpf.kernels.Matern52, gpf.kernels.White]
reduced_kernels = [gpf.kernels.SquaredExponential, gpf.kernels.Matern32,
                   gpf.kernels.RationalQuadratic, gpf.kernels.Exponential, gpf.kernels.Linear, gpf.kernels.Polynomial]
for L in range(1, 4):
    for K_L in reduced_kernels:
        for M_F in [None, gpf.functions.Polynomial(2)]:
            m, BIC = fit_model(X_aug, Y_aug, P=3, L=L, K_L=K_L, M_F=M_F)
            if BIC < best_BIC:
                best_model = m
                best_BIC = BIC
                best_L = L
                best_K_L = K_L
                best_M_F = M_F

# %%
print(best_BIC)
if 'best_L' in locals():
    print(best_L)
else:
    print("best_L is not defined")
print(best_K_L.__name__)
print(best_M_F.__class__.__name__)
print_summary(best_model)
count_params(m)


# %%
# Trying different kernels

# also known as the rbf kernel
def gaussain_kernel(x, y, s, l):
    k = (s**2)*np.exp(-(np.subtract.outer(x, y)**2)/(2*l**2))
    return k


def anova_kernel(x, y, s, d):
    try:
        k = np.exp(-(np.subtract.outer(x, y)**2)*s)**d
        for i in range(1, x.shape[1]):
            k += np.exp(-(np.subtract.outer(x[:, i], y[:, i])**2)*s)**d
    except:
        k = np.exp(-(np.subtract.outer(x, y)**2)*s)**d
    return k


def wave_kernel(x, y, t, d):
    th = np.abs(np.subtract.outer(x, y))/t
    k = (1/(th+d))*np.sin(th)

    return k

# make the wave kernel a GPFlow kernel


class Wave(gpf.kernels.Kernel):
    def __init__(self, t, d, active_dims=None):
        super().__init__(active_dims=active_dims)
        self.t = gpf.Param(t, transform=gpf.transforms.positive)
        self.d = gpf.Param(d, transform=gpf.transforms.positive)

    def K(self, X, X2=None):
        if X2 is None:
            X2 = X
        th = np.abs(np.subtract.outer(X[:, 0], X2[:, 0]))/self.t
        k = (1/(th+self.d))*np.sin(th)
        return k

    def Kdiag(self, X):
        return np.diag(self.K(X))


class CustomAnovaKernel(gpf.kernels.Kernel):
    def __init__(self, scale=1.0, degree=1, **kwargs):
        super().__init__(**kwargs)
        self.scale = gpf.Parameter(scale, transform=gpf.utilities.positive())
        self.degree = gpf.Parameter(degree, transform=gpf.utilities.positive())

    def K(self, X, X2=None):
        """
        Compute the covariance matrix between X and X2 with the ANOVA kernel function.
        """
        if X2 is None:
            X2 = X

        # Initialize kernel with zeros
        k = tf.zeros((X.shape[0], X2.shape[0]), dtype=gpf.default_float())

        for i in range(X.shape[1]):
            # Distance for dimension i
            dist = (X[:, i:i+1] - X2[:, i:i+1].T)**2

            # ANOVA kernel for dimension i
            k_i = tf.exp(-self.scale * dist) ** self.degree

            # Add to total kernel
            k += k_i

        return k

    def K_diag(self, X):
        """
        Compute the diagonal of the covariance matrix of X with the ANOVA kernel function.
        """
        return tf.fill(tf.shape(X)[:-1], tf.squeeze(self.scale ** self.degree))


def log_kernel(x, y, d):
    k = - np.log(np.abs(np.subtract.outer(x, y))**d + 1)
    return k


def cauchy_kernel(x, y, s):
    k = 1/(1+(np.subtract.outer(x, y))**2/s)
    return k


def generalised_student_t_kernel(x, y, d):
    k = 1/(1+np.abs(np.subtract.outer(x, y))**d)
    return k


# %%
def plot_model(m, X, P, L, K_L, M_F, BIC):
    fig, ax = plt.subplots(figsize=(15, 5), ncols=3, nrows=1)
    ax[0].plot(X[:, 0], Y[:, 0], "bx", mew=2)
    ax[1].plot(X[:, 0], Y[:, 1], "gx", mew=2)
    ax[2].plot(X[:, 0], Y[:, 2], "rx", mew=2)

    # just use the GP to predict at same timepoints
    mu1, var1 = m.predict_y(np.hstack((X, np.zeros_like(X))))
    plot_gp_d(X, mu1, var1, "b", "Y1", ax[0])

    mu2, var2 = m.predict_y(np.hstack((X, np.ones_like(X))))
    plot_gp_d(X, mu2, var2, "g", "Y2", ax[1])

    mu3, var3 = m.predict_y(np.hstack((X, 2*np.ones_like(X))))
    plot_gp_d(X, mu3, var3, "r", "Y3", ax[2])

    fig.suptitle('species= ' + str(P) + ', latent_processes= ' + str(L) + ', kernel= ' +
                 str(K_L) + ', mean= ' + str(M_F.__class__.__name__) + ', BIC =' + str(BIC))


def fit_model(X_aug, Y_aug, P, L, K_L=gpf.kernels.SquaredExponential, M_F=None):

    # Base kernel for leatent processes
    # k = gpf.kernels.Matern32(active_dims=[0])
    # k = gpf.kernels.SquaredExponential(active_dims=[0])

    k = K_L

    # Coregion kernel
    coreg = gpf.kernels.Coregion(
        output_dim=P,
        rank=L,
        active_dims=[1]
    )

    kern = k * coreg

    # This likelihood switches between Gaussian noise with different variances for each f_i:
    lik = gpf.likelihoods.SwitchedLikelihood(
        [gpf.likelihoods.Gaussian() for _ in range(P)]
    )

    # now build the GP model as normal
    m = gpf.models.VGP((X_aug, Y_aug), kernel=kern,
                       likelihood=lik, mean_function=M_F)

    # fit the covariance function parameters
    maxiter = reduce_in_tests(10000)
    res = gpf.optimizers.Scipy().minimize(
        m.training_loss,
        m.trainable_variables,
        options=dict(maxiter=maxiter),
        method="L-BFGS-B",
    )

    print_summary(m)
    BIC = get_BIC(m, res.fun, X.shape[0])
    print("BIC:", BIC)
    plot_model(m, X, P, L, K_L, M_F, BIC)

    return m, BIC

# %%
# fit a GP model with latent process = 1 and mean function = None
# using the gaussian kernel, anova kernel, wave kernel, log kernel, cauchy kernel and generalised student t kernel


# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=gaussain_kernel, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=anova_kernel, M_F=None)
anova_kernel = CustomAnovaKernel()
m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=anova_kernel, M_F=None)
print_summary(anova_kernel)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=CustomAnovaKernel(), M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=log_kernel, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=cauchy_kernel, M_F=None)
# m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=generalised_student_t_kernel, M_F=None)

# Identifying the best kernel, mean function and latent process dimensionality for the data set using BIC score as the metric for comparison of models

best_BIC = 10000000

# %% Testing the ANOVA kernel
# You can create ANOVA kernels by using the build in sum and product of existing kernels (say Bias and RBF) with an active_dimension for the RBF . With this approach you don't have to write your own kernel class.
# The following code creates an ANOVA kernel with 2 dimensions, one for the Bias and one for the RBF kernel.

bias = gpf.kernels.Bias()
k0 = gpf.kernels.RBF(active_dims=[0])
k1 = gpf.kernels.RBF(active_dims=[1])
kernel = (bias + k0) * (bias + k1)

m, BIC = fit_model(X_aug, Y_aug, P=3, L=1, K_L=kernel, M_F=None)
print_summary(kernel)

# %%
