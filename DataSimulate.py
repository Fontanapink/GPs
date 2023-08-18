# This script will be used to simulate the data that will be used to fit the Gaussian Process

# Very simple script that is replaced by the gLVM model script

# Import the required libraries
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# Define the function that will simulate the data
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


# Define the initial conditions
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

# plot the data
fig, ax = plt.subplots(figsize=(15, 5), ncols=3, nrows=1)
ax[0].plot(ts, y_hat[0, :], "bx", mew=2)
ax[1].plot(ts, y_hat[1, :], "gx", mew=2)
ax[2].plot(ts, y_hat[2, :], "rx", mew=2)

# show the plot
plt.show()

# export the data as a csv file
np.savetxt("data.csv", np.vstack((ts, y_hat)).T, delimiter=",", header="t,y1,y2,y3", comments="")

