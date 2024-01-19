### /Users/cbarnes-home/dev/miniconda/envs/pymc_env/bin/python3 run_sim_var.py

import os
import numpy as np
import pymc as pm
import pytensor.tensor as at
import pandas as pd
import matplotlib.pyplot as plt
import arviz as az
import seaborn as sns

from pytensor.printing import Print

def make_plot_overlay(dataX, dataS):
    nX = len(dataX[0])  # Number of columns in dataX
    nS = len(dataS[0])  # Number of columns in dataS

    fig, axs = plt.subplots(2, 1, figsize=(10, 4))  # Create a figure with two subplots

    # Plot for dataX
    for i in range(nX):
        axs[0].plot(dataX[:, i], label="X" + str(i))
    axs[0].set_title("Abundance, X")
    #axs[0].legend()

    # Plot for dataS
    for i in range(nS):
        axs[1].plot(dataS[:, i], label="S" + str(i))
    axs[1].set_title("Metabolites, S")
    #axs[1].legend()

    plt.tight_layout()  # Adjust the layout
    plt.savefig("plot-data-XS-overlay.pdf")

def make_plot_stacked(dataX, dataS):
    # add 5.6 to add abundance data
    dataX = dataX + 1.0
    
    # stacked 
    nX = len(dataX[0])  # Number of columns in dataX
    nS = len(dataS[0])  # Number of columns in dataS
    nobs = dataS.shape[0]

    fig, axs = plt.subplots(2, 1, figsize=(10, 4))  # Create a figure with two subplots
    ## Stack plot for dataX
    axs[0].stackplot(range(len(dataX)), *dataX.T, labels=["X" + str(i) for i in range(nX)])
    axs[0].set_title("Abundance, log10 X")
    axs[0].set_ylabel("X")
    axs[0].set_xlim(0,nobs-1)
    #axs[0].legend(loc='upper left')

    # Stack plot for dataS
    #axs[1].stackplot(range(len(dataS)), *dataS.T, labels=["S" + str(i) for i in range(nS)])
    #axs[1].set_title("Metabolites, S")
    #axs[1].legend(loc='upper left')

    # Heatmap for dataS
    #sns.heatmap(dataS, annot=False, cmap="YlGnBu", xticklabels=range(1, dataS.shape[0] + 1), yticklabels=["S" + str(i) for i in range(nS)], ax=axs[1])
    sns.heatmap(dataS.T, annot=False, cmap="YlGnBu", yticklabels=["S" + str(i) for i in range(nS)], ax=axs[1], cbar=False)
    axs[1].set_title("Metabolites, S")
    axs[1].set_ylabel("S")
    axs[1].set_xlabel("time (weeks)")
    axs[1].set_xlim(0,nobs)

    # Align x-axes of top and bottom figures
    #axs[0].get_shared_x_axes().join(axs[0], axs[1])
    #axs[1].set_xticklabels(axs[1].get_xticks())

    plt.tight_layout()  # Adjust the layout
    plt.savefig("plot-data-XS-stacked.pdf")

def make_plot(dataX, dataS):
    nX = len(dataX[0])  # Number of columns in dataX
    nS = len(dataS[0])  # Number of columns in dataS

    fig, axs = plt.subplots(nX + nS, 1, figsize=(10, 2*(nX+nS)))
    for i, ax in enumerate(axs):
        if i< nX:
            axs[i].plot(dataX[:,i] )
            axs[i].set_title( "X"+str(i) )
        else:
            axs[i].plot(dataS[:,i-nX] )
            axs[i].set_title( "S"+str(i-nX) )
    plt.savefig("plot-data-XS.pdf")

def generate_mvar1_data(n_obs, coefficients, coefficientsM, initial_values, initial_valuesM, noise_stddev=1):
    
    nX = len(initial_values)
    data = np.zeros((n_obs, nX))
    data[0, :] = initial_values[:,0]

    nS = len(initial_valuesM)
    dataM = np.zeros((n_obs, nS))
    dataM[0, :] = initial_valuesM[:,0]
    
    for t in range(1, n_obs):
        # VAR(1) process: X_t = A * X_{t-1} + noise
        noise = np.random.normal(scale=noise_stddev, size=nX)
        #print("A", coefficients.shape)
        #print("X", data[t - 1, :].shape)
        data[t, :] = np.dot(coefficients, data[t - 1, :]) + noise

    for t in range(1, n_obs):
        # process: S_t = B * X_{t-1} + noise
        noise = np.random.normal(scale=noise_stddev, size=(nS))
        #print("B:", coefficientsM.shape)
        #print("X", data[t - 1, :].shape)
        
        Xt = data[t - 1, :].reshape( (nX,1) )
        #print( "mult:", (coefficientsM @ Xt).shape )
        product = coefficientsM @ Xt
        dataM[t, :] = product[:,0] + noise

    return data, dataM

def simple_sim():
    # Example usage:
    np.random.seed(42)  # for reproducibility
    n_obs = 96
    nX = 2
    A = np.array([[0.8, -0.2], [0.3, 0.5]])
    X0 = np.array([1, 2]).reshape( (2,1) )

    nS = 3
    #B = np.array([[0, 0], [0.3, 0.5]])

    B = np.zeros((nS,nX))
    B[0,1] = 0.8
    B[2,1] = -0.5
    print(B)

    S0 = np.zeros(nS).reshape( (nS,1) )
    print(S0)

    dataX, dataS = generate_mvar1_data(n_obs, A, B, X0, S0)

    fig, axs = plt.subplots(nX + nS, 1, figsize=(10, 2*(nX+nS)))
    for i, ax in enumerate(axs):
        if i< nX:
            axs[i].plot(dataX[:,i] )
            axs[i].set_title( "X"+str(i) )
        else:
            axs[i].plot(dataS[:,i-nX] )
            axs[i].set_title( "S"+str(i-nX) )
    plt.savefig("plot-data-XS.pdf")

def run_inference():
    n_obs = 96
    nX = 2
    A = np.array([[0.8, -0.2], [0.3, 0.5]])
    print(A)
    X0 = np.array([1, 2]).reshape( (2,1) )

    nS = 3
    #B = np.array([[0, 0], [0.3, 0.5]])

    B = np.zeros((nS,nX))
    B[0,1] = 0.8
    B[2,1] = -0.5
    print(B)

    S0 = np.zeros(nS).reshape( (nS,1) )
    #print(S0)

    dataX, dataS = generate_mvar1_data(n_obs, A, B, X0, S0)

    #make_plot(dataX, dataS)
    #make_plot_stacked(dataX, dataS)
    #make_plot_overlay(dataX, dataS)

    # PyMC3 model
    with pm.Model() as var_model:
        # Priors for x0 and sigma
        X0h = pm.Normal('X0h', mu=X0, sigma=1, shape=(nX,1))
        S0h = pm.Normal('S0h', mu=S0, sigma=1, shape=(nS,1))
        Ah = pm.Normal('Ah', mu=0, sigma=1, shape=(nX, nX))
        Bh = pm.Normal('Bh', mu=0, sigma=1, shape=(nS, nX))

        sigma = pm.HalfNormal('sigma', sigma=1, shape=(nX+nS))

        print( "dataX:", dataX.shape)
        print( "dataS:", dataS.shape)
        data = np.concatenate( (dataX, dataS), axis=1 )
        print( "data:", data.shape)

        if 0:
            ## S and X decoupled
            muX = pm.Deterministic( 'muX', pm.math.dot(Ah, dataX[:-1, :].T) )  
            print( "muX:", muX.shape)

            muS = pm.math.dot(Bh, dataX[:-1, :].T)  
            ##muX_T = muX.T
            #print( "muX_T", muX_T.shape)
            #muS = S0h + pm.math.dot(Bh, muX.T)

            print( "muS:", muS.shape)

            mu = pm.math.concatenate( (muX,muS), axis=0)
            print( "mu:", mu.T.shape)

            likelihood = pm.Normal('likelihood', mu=mu.T, sigma=sigma, observed=data[1:, :])
        else:
            ## S and X coupled
            muX = pm.Deterministic( 'muX', pm.math.dot(Ah, dataX[:-1, :].T) )  

            print( "muX:", muX.shape)
            #muX_T = muX.T
            #muS = pm.math.dot(Bh, muX_T[:-1,:])
            muS = pm.math.dot(Bh, muX)
            print( "muS:", muS.shape)  

            muXs = muX[:,1:]
            muSs = muS[:,:-1]

            mu = pm.math.concatenate( (muXs,muSs), axis=0)
            print( "mu:", mu.T.shape)

            likelihood = pm.Normal('likelihood', mu=mu.T, sigma=sigma, observed=data[2:, :])

    # Sampling from the posterior
    with var_model:
        idata = pm.sample(2000, tune=1000, cores=2)

    print( az.summary(idata, var_names=["Ah","Bh"]) )

    # write data out
    az.to_netcdf(idata, 'model_posterior.nc')
    np.savez("params.npz", A=A, B=B)
    np.savez("data.npz", dataX=dataX, dataS=dataS)

def run_inference_large():

    n_obs = 96
    nX = 8
    X0 = np.ones([nX]).reshape( (nX,1) )*0.1
    A = np.zeros((nX,nX))
    
    A[2,3] = 0.6
    A[1,4] = -0.9
    A[0,2] = 0.8
    A[5,7] = -0.8
    A[6,1] = 0.7
    print(A)
    
    nS = 10
    S0 = np.zeros(nS).reshape( (nS,1) )
    B = np.zeros((nS,nX))
    B[0,1] = 0.8
    B[2,1] = -0.9
    B[3,6] = 0.9
    B[8,5] = -0.9
    print(B.T)

    noise_stddev = 0.1
    dataX, dataS = generate_mvar1_data(n_obs, A, B, X0, S0, noise_stddev)

    #make_plot(dataX, dataS)
    #make_plot_stacked(dataX, dataS)
    #make_plot_overlay(dataX, dataS)

    # Params for shrinkage
    DA = nX*nX
    DA0 = 5
    DB = nS*nS
    DB0 = 4
    N = n_obs - 2

    # create and fit PyMC model
    with pm.Model() as var_model:
        tau0_A = (DA0 / (DA - DA0)) * noise_stddev / np.sqrt(N)
        c2_A = pm.InverseGamma("c2_A", 2, 1)
        tau_A = pm.HalfCauchy("tau_A", beta=tau0_A)
        lam_A = pm.HalfCauchy("lam_A", beta=1, shape=(nX, nX) )
        Ah = pm.Normal('Ah', mu=0, sigma = tau_A * lam_A*at.sqrt(c2_A / (c2_A + tau_A**2 * lam_A**2)), shape=(nX, nX) )

        tau0_B = (DA0 / (DA - DA0)) * noise_stddev / np.sqrt(N)
        c2_B = pm.InverseGamma("c2_B", 2, 1)
        tau_B = pm.HalfCauchy("tau_B", beta=tau0_B)
        lam_B = pm.HalfCauchy("lam_B", beta=1, shape=(nS, nX) )
        Bh = pm.Normal('Bh', mu=0, sigma = tau_B * lam_B*at.sqrt(c2_B / (c2_B + tau_B**2 * lam_B**2)), shape=(nS, nX) )

        sigma = pm.TruncatedNormal('sigma', mu=0.1, sigma=0.1, lower=0, shape=(nX+nS))
        #sigma = 0.1

        data = np.concatenate( (dataX, dataS), axis=1 )

        muX = pm.Deterministic( 'muX', pm.math.dot(Ah, dataX[:-1, :].T) )  
        muS = pm.math.dot(Bh, muX)
        muXs = muX[:,1:]
        muSs = muS[:,:-1]
        mu = pm.math.concatenate( (muXs,muSs), axis=0)
        likelihood = pm.Normal('likelihood', mu=mu.T, sigma=sigma, observed=data[2:, :])

        #muX = pm.Deterministic( 'muX', pm.math.dot(Ah, dataX[:-1, :].T) ) 
        #muS = pm.Deterministic( 'muS', pm.math.dot(Bh, muX) )
        #mu = pm.math.concatenate( (muX,muS), axis=0)
        #likelihood = pm.Normal('likelihood', mu=mu.T, sigma=sigma, observed=data[1:, :])

        # Sampling from the posterior
    with var_model:
        trace = pm.sample(1000, tune=3000, cores=4)

    print( az.summary(trace, var_names=["Ah","Bh"]) )

    # write data out
    az.to_netcdf(trace, 'model_posterior.nc')
    np.savez("params.npz", A=A, B=B)
    np.savez("data.npz", dataX=dataX, dataS=dataS)

def posterior_analysis(simulated=True):

    # first plot the data:
    with np.load("data.npz", allow_pickle=True) as xsdata:
        dataX = xsdata['dataX']
        dataS = xsdata['dataS']
    make_plot_stacked(dataX, dataS)

    idata = az.from_netcdf('model_posterior.nc')

    print( az.summary(idata, var_names=["Ah","Bh"]) )

    if simulated == True:
        with np.load("params.npz", allow_pickle=True) as pdata:
            A = pdata['A']
            B = pdata['B']

        az.plot_posterior(idata, var_names=["Ah"], ref_val=A.flatten().tolist() )
        plt.savefig("plot-posterior-Ah.pdf")
        az.plot_posterior(idata, var_names=["Bh"], ref_val=B.flatten().tolist() )
        plt.savefig("plot-posterior-Bh.pdf")

    else:
        az.plot_posterior(idata, var_names=["Ah"], ref_val=A.flatten().tolist() )
        plt.savefig("plot-posterior-Ah.pdf")
        az.plot_posterior(idata, var_names=["Bh"], ref_val=B.flatten().tolist() )
        plt.savefig("plot-posterior-Bh.pdf")


    # Make heatmaps
    matrix1 = idata.posterior['Ah'].values
    matrix2 = idata.posterior['Bh'].values

    # Assuming the matrices have more than one dimension and you are interested in specific ones
    # E.g., taking the mean over the first dimension (chain) if it exists
    matrix1_sum = np.median(matrix1, axis=(0, 1))
    matrix2_sum = np.median(matrix2, axis=(0, 1))

    # Create subplots
    fig, ax = plt.subplots(1, 2, figsize=(14, 7),  gridspec_kw={'width_ratios': [1, 1.2]})

    # Heatmap for matrix1
    sns.heatmap(matrix1_sum, ax=ax[0], cmap='viridis')
    ax[0].set_title('Ahat')
    ax[0].set_ylabel('X')
    ax[0].set_xlabel('X')

    # Annotate the true values for matrix1
    for i in range(matrix1_sum.shape[0]):
        for j in range(matrix1_sum.shape[1]):
            ax[0].text(j + 0.5, i + 0.5, f'{A[i, j]:.2f}', ha='center', va='center', color='white')

    # Heatmap for matrix2
    matrix2_sum = matrix2_sum.T 
    BT = B.T
    sns.heatmap(matrix2_sum, ax=ax[1], cmap='viridis')
    ax[1].set_title('Bhat')
    ax[1].set_xlabel('S')

    # Annotate the true values for matrix2
    for i in range(matrix2_sum.shape[0]):
        for j in range(matrix2_sum.shape[1]):
            ax[1].text(j + 0.5, i + 0.5, f'{BT[i, j]:.2f}', ha='center', va='center', color='white')

    plt.savefig('plot-posterior-heatmap.pdf', bbox_inches='tight')


if __name__ == '__main__':
    #simple_sim()
    #run_inference()
    
    run_inference_large()
    posterior_analysis()