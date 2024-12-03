import numpy as np
import matplotlib.pyplot as plt

# n chosen based on problem 3.5 in the homework
n = 40000

# Initializing the vector using np.random.normal function
Zs = np.random.normal(0, 1, n)
k_values = [1, 8, 64, 512]

# Defining ecdf function
def ecdf(x, samples): 
    return np.mean(samples <= x)

# Plotting the graph on the appropriate range
x_values = np.linspace(-3, 3, 1000)

# Running the ecdf function on the vector of sampled normal r.vs and plotting the results
ecdf_values = [ecdf(x, Zs) for x in x_values]
plt.plot(x_values, ecdf_values, label='ECDF (N(0, 1))')

# This is part b, here we're calculating the ECDF of the Y_k for different k values
for k in k_values:
    Y_k = np.sum(np.sign(np.random.randn(n, k))*np.sqrt(1./k), axis=1)
    ecdf_Y_k = [ecdf(x, Y_k) for x in x_values]
    plt.plot(x_values, ecdf_Y_k, label=f'{k}')

plt.xlabel('Observations')
plt.ylabel('Probability')
plt.title('ECDFs of Y^(k) for different k and N(0, 1)')
plt.legend()
plt.grid(True)
plt.show()



