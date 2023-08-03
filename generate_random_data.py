import copy
import torch
import pickle
import numpy as np

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
    X_train = copy.deepcopy(X)
    Y_train = copy.deepcopy(Y)
    X_train = [torch.tensor(member) for member in X_train]
    Y_train = [torch.tensor(member) for member in Y_train]
    last_y = Y_train[-1]
    Y_train = Y_train[:-1]
    dim = len(X[0])

    variables = {
        "x_train": X_train,
        "y_train": Y_train,
        "last_y": last_y,
        "dim": dim
    }

    return variables


def pickle_data(variables, file_name, folder_name):
    # Save to a file
    file_name_pickled = folder_name + "/" + file_name + ".pkl"
    with open(file_name_pickled, "wb") as outfile:
        pickle.dump(variables, outfile)
    
for i in range(30):
    variables = generate_stdp_dataset(10, 442, 0, 1)
    base_file_name = f"dataset_{i}"
    folder_name = "random_datasets"
    pickle_data(variables, base_file_name, folder_name)

# for i in range(30):
#     with open(f'../generated_stdp_data/dataset_{i}.pkl', 'rb') as infile:
#         variables = pickle.load(infile)
#         print(f"last_y: {variables['last_y']}")
