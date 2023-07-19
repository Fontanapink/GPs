#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 14:51:00 2019
Author: Pedro Fontanarrosa

Description:
This script will fit a Gaussian Process to the imported data
and then plot the results
To do so, it will first import the data from the file
and then generate a library of all the possible kernel, mean, and latent processes combinations
It will then fit each of these combinations to the data
then it will report the best fitted model
and finally it will plot the results of the best fitted model
The script will also save the results of the best fitted model to a file
The script could also plot all the BIC values for each of the fitted models and the one chosen to be the best
The script could also plot the results of the best fitted model with the data
The script could also plot the results of the best fitted model with the data and the 95% confidence interval

usage: GPFit.py [-h] -i INPUT [-o OUTPUT]
-i <input file>: The path to the input file
-o <output file path>: The path to the output file
"""

# Import the necessary libraries
import argparse
import numpy as np
# Set the random seed for reproducibility
# FIXME: This should be removed in the final version
np.random.seed(0)
import pandas as pd
import os
import sys
import math
import matplotlib.pyplot as plt
import gpflow as gpf # This is the library that will be used to fit the Gaussian Process
from gpflow.utilities import print_summary
from gpflow.utilities import parameter_dict
from gpflow.ci_utils import reduce_in_tests
gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("notebook")

# Define the maximum number of iterations for the optimization
# FIXME: This should be an input argument
MAXITER = reduce_in_tests(5000)

import tensorflow as tf
import time
import datetime
import pickle

# Define the function that will import the data from the file
def importData(inputFile):
    # Import the data from the file
    data = pd.read_csv(inputFile, sep=',')
    # Return the data
    return data

# Define the function that will augment the data for the Gaussian Process
def augmentData(timeSeries, y):
    # Augment the input with ones or zeros to indicate the required output dimension
    X_aug = np.vstack(
        (np.hstack((timeSeries, np.zeros_like(timeSeries))),
         np.hstack((timeSeries, np.ones_like(timeSeries))),
         np.hstack((timeSeries, 2*np.ones_like(timeSeries)))
         )
    )
    # Augment the Y data with ones or zeros that specify a likelihood from the list of likelihoods
    # FIXME: make this more general, so that it can handle more than 3 outputs
    Y1 = y[:, 0].reshape(-1, 1)
    Y2 = y[:, 1].reshape(-1, 1)
    Y3 = y[:, 2].reshape(-1, 1)

    Y_aug = np.vstack(
        (np.hstack((Y1, np.zeros_like(Y1))),
         np.hstack((Y2, np.ones_like(Y2))),
         np.hstack((Y3, 2*np.ones_like(Y3)))
         )
    )
    # Return the augmented data

    return X_aug, Y_aug

# Define the function that will generate the library of all the possible kernel, mean, and latent processes combinations
def generateLibrary():
    # Define the list of all the possible kernel, mean, and latent processes
    kernelList = [gpf.kernels.SquaredExponential, gpf.kernels.Matern32, gpf.kernels.RationalQuadratic, gpf.kernels.Exponential, gpf.kernels.Linear,
           gpf.kernels.Cosine, gpf.kernels.Periodic, gpf.kernels.Polynomial, gpf.kernels.Matern12, gpf.kernels.Matern52, gpf.kernels.White]
    meanList = [gpf.mean_functions.Constant(), gpf.mean_functions.Linear(), gpf.mean_functions.Identity(), gpf.mean_functions.Zero(), 
                gpf.mean_functions.Polynomial(2), gpf.mean_functions.Polynomial(3), gpf.mean_functions.Polynomial(4), gpf.mean_functions.Polynomial(5)]
    latentList = [gpf.likelihoods.Gaussian(), gpf.likelihoods.StudentT()]
    # Define the list of all the possible combinations of kernel, mean, and latent processes
    library = []
    for kernel in kernelList:
        for mean in meanList:
            for latent in latentList:
                library.append([kernel, mean, latent])
    # Return the library
    return library

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
    p_dict = parameter_dict(m.trainable_parameters)
    # p_dict = parameter_dict(m)
    p_count = 0
    for val in p_dict.values():
        # print(val.shape)
        if len(val.shape) == 0:
            p_count = p_count + 1
        else:
            p_count = p_count + math.prod(val.shape)

    return p_count

# This is for model selection: the higher the BIC the better the model
def get_BIC(m, F, n):
    k = count_params(m)
    return -2 * F + k * np.log(n)

def main():
    parser = argparse.ArgumentParser(description='Gaussian Process fitting and plotting')
    parser.add_argument('-i','--input', help='Input file name', required=True)
    parser.add_argument('-o','--output', help='Output file path', required=False, default='output.csv')
    args = vars(parser.parse_args())

    # Store the input arguments
    inputFile = args['input']
    # Check if the input file exists, if not, exit the script with an error message
    if not os.path.isfile(inputFile):
        print('The input file does not exist')
        sys.exit()
    # Check if the input file is empty, if it is, exit the script with an error message    
    if os.path.getsize(inputFile) == 0:
        print('The input file is empty')
        sys.exit()
    # Check if the input file is a csv file, if not, exit the script with an error message    
    if not os.path.splitext(inputFile)[1] == '.csv':
        print('The input file is not a csv file')
        sys.exit()

    outputFile = args['output']
    # Check if the output file path exists, if not, set it to the current directory
    if not os.path.isdir(os.path.dirname(outputFile)):
        print('The output file path does not exist, setting it to the current directory')
        outputFile = os.path.join(os.getcwd(), os.path.basename(outputFile))

    data = importData(inputFile)
    # store the time series data in a variable, without the column header
    


    timeSeries = data.iloc[:, 0].values
    timeSeries = timeSeries.reshape(-1, 1)
    y = data.iloc[:, 1:].values
    #y = y.T
    X, Y = augmentData(timeSeries, y)

    print(X.shape)
    print(Y.shape)

if __name__ == "__main__":
    main()
