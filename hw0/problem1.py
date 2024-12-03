import numpy as np

# Initializing A and using numpy np.linalg.inv function to calculate A^-1
A = np.array([[0,2,4],[2,4,2],[3,3,1]])
A_inv = np.linalg.inv(A)

# More initialization
b = [[-2],[-2],[-4]]
c = [[1],[1],[1]]

# Using np.matmul to calculate matrix product
res_1 = np.matmul(A_inv, b)
res_2 = np.matmul(A, c)

# Printing result
print("A inverse: \n", A_inv)
print("A inverse times b: \n", res_1)
print("A times c: \n", res_2)
