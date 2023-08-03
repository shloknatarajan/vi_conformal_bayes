#!/usr/bin/env python
# coding: utf-8

# Imports
import math
import os
import torch
import torch.distributions.constraints as constraints
import pyro
import numpy as np
import copy
import time
import snakeviz
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import random
from sklearn.datasets import load_diabetes
assert pyro.__version__.startswith('1.8.4')

# clear the param store in case we're in a REPL
pyro.clear_param_store()

# Load Datasets
## Random Dataset
def generate_stdp_dataset(dim, num_examples, min_value, max_value):
    X = np.random.random((num_examples + 1, dim)) * (max_value - min_value) + min_value
    beta = np.random.random((dim)) * (max_value - min_value) + min_value

    noise = np.random.normal(0, np.sqrt(max_value - min_value), num_examples + 1)
    Y = X[:num_examples + 1] @ beta + noise

    X = np.asfortranarray(X)
    Y = np.asfortranarray(Y)
    X /= np.linalg.norm(X, axis=0)
    Y = (Y - Y.mean()) / Y.std()
    Y = Y * max_value

    Y = Y/np.linalg.norm(Y)

    return X, Y, beta

X, Y, beta = generate_stdp_dataset(3, 10, 0, 1)

## Diabetes Dataset
X, Y = load_diabetes(return_X_y = True)
Y = (Y - Y.mean()) / Y.std()
X = X / np.linalg.norm(X)

## Set up X and Y Train
X_train = copy.deepcopy(X)
Y_train = copy.deepcopy(Y[:len(Y) - 1])
X_train = [torch.tensor(member) for member in X_train]
Y_train = [torch.tensor(member) for member in Y_train]
dim = len(X[0])


# Model and Guide Setup
def model(data):
    # define the hyperparameters that control the Beta prior
    mu0 = torch.zeros(dim, dtype=torch.float64)
    # sample f from the Beta prior
    f = pyro.sample("latent_fairness", dist.MultivariateNormal(mu0, std0))
    # loop over the observed data
    subset = random.sample(data, int(len(data) / dim))
    for i in range(len(subset)):
        pyro.sample("obs_{}".format(i), dist.Normal(f.dot(data[i][0]), 0.3), obs=data[i][1])

def guide(data):
    # register the two variational parameters with Pyro
    # - both parameters will have initial value 15.0.
    # - because we invoke constraints.positive, the optimizer
    # will take gradients on the unconstrained parameters
    # (which are related to the constrained parameters by a log)
    mu_q = pyro.param("mu_q", copy.deepcopy(prev_mu_q))
    # sample latent_fairness from the distribution Beta(alpha_q, beta_q)
    pyro.sample("latent_fairness", dist.MultivariateNormal(mu_q, std0))

def train_svi(D_hat, n_steps):
    # setup the optimizer
    adam_params = {"lr": 0.005, "betas": (0.90, 0.999)}
    optimizer = Adam(adam_params)

    # setup the inference algorithm
    svi = SVI(model, guide, optimizer, loss=Trace_ELBO())

    # do gradient steps
    for step in range(n_steps):
        loss = svi.step(D_hat)
        if loss < 1e-5:
            break
    
    mu_q = pyro.param("mu_q")
    return mu_q



global prev_mu_q
prev_mu_q = torch.zeros(dim, dtype=torch.float64)
std0 = torch.eye(dim, dtype=torch.float64) * 0.3

def run_svi(X_train, Y_train, decrease_size=0.1, threshold=0.9):
    # Run SVI on the Data
    y_hat = max(Y_train)
    y_bottom = min(Y_train)
    print(y_hat)
    print(y_bottom)
    conformal_set = []
    start = time.time()
    while y_hat >= y_bottom:
        pyro.clear_param_store()
        # Create D_hat
        D_hat = list(zip(X_train[:-1], Y_train))
        D_hat.append((X_train[-1], y_hat))
        
        # Train SVI
        mu_q = train_svi(D_hat, 4)
        prev_mu_q = mu_q
        
        # Calculate rank of y_hat
        rank = [(abs(sum(D_hat[i][0] * mu_q) - D_hat[i][1]).detach().numpy()) for i in range(len(D_hat))]
        y_hat_rank = rank[-1]
        
        # Add to conformal set if in not in bottom 10 percent of probabilities
        if np.count_nonzero(y_hat_rank > rank) / len(rank) < threshold:
            conformal_set.append(copy.deepcopy(y_hat))
            print(f"{y_hat} Added")
        else:
            print(f"{y_hat} Not added")
            
        y_hat -= decrease_size
    conformal_set = [min(conformal_set), max(conformal_set)]
    end = time.time()

    print(f"Conformal Set: [{float(conformal_set[0])}, {float(conformal_set[1])}]")
    print(f"Length: {float(conformal_set[1] - conformal_set[0])}")
    print(f"Y[-1]: {Y[-1]}")
    if Y[-1] >= conformal_set[0] and Y[-1] <= conformal_set[1]:
        print("Y[-1] is covered")
    else:
        print("Y[-1] is Not covered")
    print(f"Elapsed Time: {end - start}")

run_svi(X_train, Y_train)



