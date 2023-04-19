import math
import os
import torch
from pyro.infer.autoguide.initialization import init_to_value
import torch.distributions.constraints as constraints
import pyro
from pyro import poutine
import pickle
import numpy as np
import copy
import time
from pyro.optim import Adam
from pyro.infer import SVI, Trace_ELBO
import pyro.distributions as dist
import random
assert pyro.__version__.startswith('1.8.4')
pyro.clear_param_store()
# clear the param store in case we're in a REPL


def pickle_data(variables, file_name, folder_name):
    # Save to a file
    file_name_pickled = folder_name + "/" + file_name + ".pkl"
    with open(file_name_pickled, "wb") as outfile:
        pickle.dump(variables, outfile)
        
datasets = list(range(30))
runs = 50
print("not frozen 1")
laplace_results = {
    'coverage': np.zeros((len(datasets), runs)),
    'length': np.zeros((len(datasets), runs)),
    'time': np.zeros((len(datasets), runs))
}


def svi_runner(datasets, num_runs):
    # # Load Datasets
    for dataset_num in datasets:
        infile = open(f'data/generated_stdp_data/dataset_{dataset_num}.pkl', 'rb')
        variables = pickle.load(infile)
        X_train = variables['x_train']
        Y_train = variables['y_train']
        last_y = variables['last_y']
        dim = variables['dim']
        infile.close()
        for run in range(num_runs):
            pyro.clear_param_store()
            
            def is_converged(arr):
                if len(arr) < 2:
                    return False
                # check if last values have changed by 1% or less
                return (max(arr) - min(arr)) / max(arr) < 0.01

            def model(data):
                # define the hyperparameters that control the Beta prior
                mu0 = torch.zeros(dim, dtype=torch.float64)
                bmean = torch.ones(dim, dtype=torch.float64)
                bscale = torch.ones(dim, dtype=torch.float64)
                
                std0 = torch.zeros(dim, dtype=torch.float64)
                s_mean = torch.ones(dim, dtype=torch.float64)
                s_scale = torch.ones(dim, dtype=torch.float64)
                
                b = pyro.sample("b", dist.Gamma(bmean, bscale).to_event(1))
                std_scale = pyro.sample("std_scale", dist.Gamma(s_mean, s_scale).to_event(1))
                
                # sample f from the Beta prior
                beta = pyro.sample("beta", dist.Laplace(mu0, b).to_event(1))
                std = pyro.sample("sigma", dist.Laplace(std0, std_scale).to_event(1))
                # data goes list tuple tensor
                    
                with pyro.plate("data", len(data), subsample_size=100) as ind:
                    data = [data[x] for x in ind]
                    for i in range(len(data)):
                        sampler = dist.Normal(beta.dot(data[i][0]).item(), torch.sigmoid(std.dot(data[i][0])).item())
                        pyro.sample("obs_{}".format(i), sampler.to_event(0), obs=data[i][1])

            # Global guide
            guide = pyro.infer.autoguide.AutoLaplaceApproximation(poutine.block(model, expose=['b', 'beta', 'std_scale', 'sigma']))

            def train_SVI(D_hat, n_steps, warm_dict = None):    
                losses = []
                # setup the optimizer
                adam_params = {"lr": 0.005, "betas": (0.90, 0.999)}
                optimizer = Adam(adam_params)

                # setup the inference algorithm
                guide = pyro.infer.autoguide.AutoLaplaceApproximation(poutine.block(model, expose=['b', 'beta', 'std_scale', 'sigma']))
                if warm_dict is not None:
                    guide = pyro.infer.autoguide.AutoLaplaceApproximation(poutine.block(model, expose=['b', 'beta', 'std_scale', 'sigma']), init_loc_fn=init_to_value(values=warm_dict))
                svi = SVI(model, guide, optimizer, loss=Trace_ELBO())
                # do gradient steps
                for step in range(n_steps):
                    loss = svi.step(D_hat)
                    losses.append(loss)
                    if is_converged(losses[-1:-6:-1]):
                        # print(f"Converged on step {step} -- breaking")
                        break
                    if loss < 1e-5:
                        # print(f"Early Loss Stopping on step {step}")
                        break
                beta = guide(D_hat)['beta']
                # print(f"Ended on step {step}")
                return beta, losses

            rank_proportions = []
            y_hat = max(Y_train)
            y_bottom = min(Y_train)
            conformal_set = []
            decrease_size = 0.0025
            start = time.time()
            all_losses = []
            while y_hat >= y_bottom:
                pyro.clear_param_store()
                # Create D_hat
                D_hat = list(zip(X_train[:-1], Y_train))
                D_hat.append((X_train[-1], y_hat))
                
                # Train SVI
                warm_dict = guide(D_hat)
                if warm_dict is not None:
                    beta, losses = train_SVI(D_hat, 100, warm_dict)
                else:
                    print("Warm dict was none")
                    beta, losses = train_SVI(D_hat, 100)
                all_losses.append(losses)
                
                
                # Calculate rank of y_hat
                rank = [(abs(sum(D_hat[i][0] * beta) - D_hat[i][1]).detach().numpy()) for i in range(len(D_hat))]
                y_hat_rank = rank[-1]
                
                # Add to conformal set if in not in bottom 10 percent of probabilities
                current_rank_proportion = np.count_nonzero(y_hat_rank > rank) / len(rank)
                rank_proportions.append(current_rank_proportion)
                if current_rank_proportion < 0.9:
                    conformal_set.append(copy.deepcopy(y_hat))
                    print(f"{y_hat} Added")
                else:
                    print(f"{y_hat} Not added")
                y_hat -= decrease_size
            conformal_set = [min(conformal_set), max(conformal_set)]
            end = time.time()
            conformal_set = [min(conformal_set), max(conformal_set)]
            length = conformal_set[1] - conformal_set[0]

            # add some - and + decrease_size if you want to do counts like etash is talking about
            coverage = last_y >= conformal_set[0] and last_y <= conformal_set[1]
            run_time = end - start
            laplace_results["coverage"][dataset_num][run] = coverage
            laplace_results["length"][dataset_num][run] = length
            laplace_results["time"][dataset_num][run] = run_time

            print(f"Completed ({dataset_num}, {run}). Coverage: {coverage}. Last Y: {last_y}. Conformal Set: ({conformal_set[0]}, {conformal_set[1]}). Time: {run_time}")
print("not frozen 2")
svi_runner(datasets, runs)
# variable, filename, folder
pickle_data(laplace_results, "laplace_std_results_4_15", "../run_saves")