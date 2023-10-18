#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:00:00 2023
Author: Pedro Fontanarrosa

Description:
This script is the same as GPFitScriptINDPENDENT.py but uses GPR instead of VGP
and then sample the posteriors and calculate the correlation between the three species.
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
-i <input file>: The path to the input file with the data
-o <output file path>: The path to the output file where the results will be stored
-m <maxiter>: The maximum number of iterations for the optimization
"""

# Import the necessary libraries
import argparse
import numpy as np
import pretty_errors
# Set the random seed for reproducibility
# FIXME: This should be removed in the final version
np.random.seed(0)
import pandas as pd
import os
import sys
import math
import matplotlib.pyplot as plt
import gpflow as gpf # This is the library that will be used to fit the Gaussian Process
#from gpflow.utilities import print_summary
from gpflow.utilities import parameter_dict
from gpflow.ci_utils import reduce_in_tests
gpf.config.set_default_float(np.float64)
gpf.config.set_default_summary_fmt("notebook")
import csv

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
        (np.hstack((timeSeries, np.zeros_like(timeSeries))))
         )
    
    # Augment the Y data with ones or zeros that specify a likelihood from the list of likelihoods
    Y1 = y.reshape(-1, 1)

    Y_aug = np.vstack(
        (np.hstack((Y1, np.zeros_like(Y1)))
         )
    )
    # Return the augmented data

    return X_aug, Y_aug

# Define the function that will generate the library of all the possible kernel, mean, and latent processes combinations
def generateLibrary():
    # Define the list of all the possible kernel, mean, and latent processes
    kernelList = [gpf.kernels.SquaredExponential, gpf.kernels.Matern32, gpf.kernels.RationalQuadratic, gpf.kernels.Exponential, gpf.kernels.Linear,
           gpf.kernels.Cosine, gpf.kernels.Periodic, gpf.kernels.Polynomial, gpf.kernels.Matern12, gpf.kernels.Matern52, gpf.kernels.White]
    # QUESTION looks like the Perdiodic kernel doesn't have active dims, then it breaks the code
    # TODO: check active dims for the periodic kernel
    kernelList = [gpf.kernels.SquaredExponential, gpf.kernels.Matern32, gpf.kernels.RationalQuadratic, gpf.kernels.Exponential, gpf.kernels.Linear,
           gpf.kernels.Cosine, gpf.kernels.Polynomial, gpf.kernels.Matern12, gpf.kernels.Matern52, gpf.kernels.White]
    # QUESTION: Is mean zero the same as no mean? or None?
    # NOTE: Decided to only try with the zero mean function
    # meanList = [gpf.mean_functions.Constant(), gpf.mean_functions.Linear(), gpf.mean_functions.Identity(), gpf.mean_functions.Zero(), 
    #             gpf.mean_functions.Polynomial(2), gpf.mean_functions.Polynomial(3), gpf.mean_functions.Polynomial(4), gpf.mean_functions.Polynomial(5)]
    mean = gpf.mean_functions.Zero()
 
    # Define the list of all the possible combinations of kernel, mean, and latent processes
    library = []
    for kernel in kernelList:
        library.append([kernel, mean])
    # Return the library
    return library

# This function is used to plot the results of the fitted model and its 95% confidence interval
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

# This function is used to plot the results of the fitted model and the data
def plot_model(m, X, Y, P, K_L, M_F, BIC, i, outputFolderPath):
    # plot the data, only one plot for each species
    fig, ax = plt.subplots(figsize=(15, 5), ncols=1, nrows=1)
    # FIXME: this should be a loop over the number of species once that is automatic
    ax.plot(X, Y, "bx", mew=2)

    color1 = "#21d524"

    # just use the GP to predict at same timepoints
    mu1, var1 = m.predict_y(X)
    # save the prediced mu1 values to a csv file in the inputfilename folder
    np.savetxt(os.path.join(outputFolderPath, 'Y_' + str(i) + 'mu1.csv'), mu1, delimiter=',')



    plot_gp_d(X, mu1, var1, "r", "Y1", ax)
    ax.set_xlabel("time")
    ax.set_ylabel("y")


    fig.suptitle('species= ' + str(P) + ', kernel= ' +
                 str(K_L.__name__) + ', mean= ' + str(M_F.__class__.__name__) + ', BIC =' + str(BIC))


# This function is used to optimize the model with scipy
# QUESTION: Should we use lmfit instead of scipy?
def optimize_model_with_scipy(model, X, Y):
    optimizer = gpf.optimizers.Scipy()
    res = optimizer.minimize(
        model.training_loss_closure(),
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
        if len(val.shape) == 0:
            p_count = p_count + 1
        else:
            p_count = p_count + math.prod(val.shape)

    return p_count

# This is for model selection: the higher the BIC the better the model
def get_BIC(m, F, n):
    k = count_params(m)
    return -2 * F + k * np.log(n)

def fit_model(X_aug, Y_aug, P, K_L=gpf.kernels.SquaredExponential, M_F=None):

    # Base kernel for leatent processes
    # k = gpf.kernels.Matern32(active_dims=[0])
    # k = gpf.kernels.SquaredExponential(active_dims=[0])

    k = K_L(active_dims=[0])

    kern = k

    # now build the GP model as normal using GPR instead of VGP
    m = gpf.models.GPR((X_aug, Y_aug), kernel=kern, mean_function=M_F)
    #m = gpf.models.VGP((X_aug, Y_aug), kernel=kern,
    #                   likelihood=lik, mean_function=M_F)


    res = optimize_model_with_scipy(m, X_aug, Y_aug)


    BIC = get_BIC(m, res.fun, X_aug.shape[0])
    print("BIC:", BIC)
    

    return m, BIC

# Create a function that will create shift the data by a given number of time steps
# For an autoregressive model, the input data is the observed value with `shift` steps of delay
# So, the X data is the observed value with 
def shiftData(y_data, shift):
    # copy x_data and y_data into a dataframe
    data = pd.DataFrame(y_data)

    # Input data is observed valued with 2 steps of delay
    X = pd.concat([data.shift(-2), data.shift(-1), data], axis=1).set_axis([0,1,2],axis=1).iloc[:-3]
 
    # Output
    Y = data.shift(-3).iloc[:-3]
 
    # Turn into Numpy arrays
    X = X.values
    Y = Y.values.reshape(-1,1)

    return X, Y

def fit_GP_models(data, library, inputFileName, outputFolderPath, showgraphs, writer):


    # P and L are the number of species and latent processes respectively
    # FIXME: These should be input arguments or obtained from the input data
    P = 1

    # store the time series data in a variable, without the column header
    timeSeries = data.iloc[:, 0].values
    timeSeries = timeSeries.reshape(-1, 1)

    # for each species in y, fit a GP
    i = 1
    for y in data.iloc[:, 1:].values.T:
        #X, Y = augmentData(timeSeries, y)
        X, Y = shiftData(y, 3)

        # Fit each of the models in the library to the data
        # and store the results in a list
        results = []
        for kernel, mean in library:
            print("kernel: ", str(kernel.__name__), ", mean: ", str(mean.__class__.__name__))
            m, BIC = fit_model(X, Y, P, kernel, mean)
            results.append([m, BIC, kernel, mean])


        # Find the best fitted model
        # by finding the model with the highest BIC
        bestModel = max(results, key=lambda x: x[1])
        m = bestModel[0]
        BIC = bestModel[1]
        K_L = bestModel[2]
        M_F = bestModel[3]

        # Write the results to the output file
        writer.writerow([inputFileName, f'Species {i}', BIC, K_L.__name__, M_F.__class__.__name__])

        # Finally, plot the model
        plot_model(m, X, Y, P, K_L, M_F, BIC, i, outputFolderPath)
        if showgraphs:
            plt.show()
        # Save the figure to a file in the current directory
        plt.savefig(os.path.join(outputFolderPath, 'Y_' + str(i) + 'plot.png'), dpi=500)
        plt.close()


        # Show a plot with all the calculated BIC values
        # and the one chosen to be the best
        BICs = [x[1] for x in results]
        plt.plot(BICs)
        plt.plot(BICs.index(BIC), BIC, 'ro')
        plt.xlabel('Model')
        plt.ylabel('BIC')

        if showgraphs:
            plt.show()
        # Save the figure to a file in the current directory, in a new folder called 'output'
        plt.savefig(os.path.join(outputFolderPath, 'Y_' + str(i) + 'BICs.png'), dpi=500)
        plt.close()

        i+=1


def main():
    # Generate the library of all the possible kernel, mean, and latent processes combinations
    library = generateLibrary()

    # Parse the input arguments
    parser = argparse.ArgumentParser(description='Gaussian Process fitting and plotting script')
    parser.add_argument('-i','--input', help='Input file name/path with simulated data', required=True)
    parser.add_argument('-o','--output', help='Output file path where output files/models are going to be stored', required=False, default='output.csv')
    # make maxiter an optional input argument, with a default value of 5000
    parser.add_argument('-m','--maxiter', help='Maximum number of iterations for the optimization', required=False, default=10000)
    # make showgraphs an optional input argument, with a default value of 'No'
    parser.add_argument('-g','--showgraphs', help='Show graphs', required=False, action='store_true')
    args = vars(parser.parse_args())

    # Store the maximum number of iterations for the optimization
    global MAXITER 
    MAXITER = reduce_in_tests(args['maxiter'])

    # Store the output file path
    outputFile = args['output']
    # Check if the output file path exists, if not, set it to the current directory
    if not os.path.isdir(os.path.dirname(outputFile)):
        print('The output file path does not exist, setting it to the current directory')
        outputFile = os.path.join(os.getcwd(), os.path.basename(outputFile))

    # Check if the 'outputs' folder exists, else create it
    outputsFolder = os.path.join(outputFile, 'outputs_auto_regressive')
    if not os.path.isdir(outputsFolder):
        os.mkdir(outputsFolder)

    # Store the input arguments
    inputFile = args['input']
    # Check if the input file exists, if not, exit the script with an error message
    if not os.path.exists(inputFile):
        print('The input file or folder does not exist')
        sys.exit()

    # Check if the input is a file or a folder
    if os.path.isfile(inputFile):
        # Check if the input file is empty, if it is, exit the script with an error message    
        if os.path.getsize(inputFile) == 0:
            print('The input file is empty')
            sys.exit()
        # Check if the input file is a csv file, if not, exit the script with an error message    
        if not os.path.splitext(inputFile)[1] == '.csv':
            print('The input file is not a csv file')
            sys.exit()

        # Store the input filename without the extension
        inputFileName = os.path.splitext(os.path.basename(inputFile))[0]

        # Check if in the outputs folder there is a folder with the same name as the input file, else create it
        outputFolderPath = os.path.join(outputsFolder, inputFileName)
        if not os.path.isdir(outputFolderPath):
            os.mkdir(outputFolderPath)

        # Import the data from the file
        data = importData(inputFile)

        output_file = os.path.join(outputFolderPath, 'output.csv')

        with open(output_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(['Input File', 'Species', 'BIC', 'Kernel', 'Mean'])

            # Fit the GP models to the data
            fit_GP_models(data, library, inputFileName, outputFolderPath, showgraphs=args['showgraphs'], writer=writer)
    
    elif os.path.isdir(inputFile):
        output_file = os.path.join(outputsFolder, 'output.csv')
        with open(output_file, 'a') as file:
            writer = csv.writer(file)
            writer.writerow(['Input File', 'Species', 'BIC', 'Kernel', 'Mean'])
            # Loop through all the csv files in the folder
            for file in os.listdir(inputFile):
                if file.endswith(".csv"):
                    # Store the input filename without the extension
                    inputFileName = os.path.splitext(os.path.basename(file))[0]

                    # Check if in the outputs folder there is a folder with the same name as the input file, else create it
                    outputFolderPath = os.path.join(outputsFolder, inputFileName)
                    if not os.path.isdir(outputFolderPath):
                        os.mkdir(outputFolderPath)

                    # Import the data from the file
                    data = importData(os.path.join(inputFile, file))

                    # Fit the GP models to the data
                    fit_GP_models(data, library, inputFileName, outputFolderPath, showgraphs=args['showgraphs'], writer=writer)
    
    else:
        print('The input is not a file or a folder')
        sys.exit()



if __name__ == "__main__":
    main()
