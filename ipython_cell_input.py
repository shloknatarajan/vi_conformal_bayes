X_train = copy.deepcopy(X)
Y_train = copy.deepcopy(Y[:len(Y) - 1])
X_train = [torch.tensor(member) for member in X_train]
Y_train = [torch.tensor(member) for member in Y_train]
y_hat = max(Y_train)
y_bottom = min(Y_train)
print(y_hat)
print(y_bottom)
conformal_set = []
step_size = 0.1
start = time.time()
while y_hat >= y_bottom:
    pyro.clear_param_store()
    # Create D_hat
    D_hat = list(zip(X_train[:-1], Y_train))
    D_hat.append((X_train[-1], y_hat))
    
    # Train SVI
    mu_q = train_SVI(D_hat, 10)
    
    # Calculate rank of y_hat
    rank = [(sum(D_hat[i][0] * mu_q) - D_hat[i][1]).detach().numpy() for i in range(len(D_hat))]
    y_hat_rank = rank[-1]
    
    # Add to conformal set if in not in bottom 10 percent of probabilities
    if np.count_nonzero(y_hat_rank > rank) / len(rank) > 0.1:
        conformal_set.append(copy.deepcopy(y_hat))
    else:
        print(f"{y_hat} Not added")
        
    y_hat -= step_size
conformal_set = [min(conformal_set), max(conformal_set)]
end = time.time()
