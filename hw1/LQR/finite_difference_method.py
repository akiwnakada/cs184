import numpy as np


def gradient(f, x, delta=0.01):
    """
    Returns the gradient of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method


    Returns:
        ret (numpy.array): gradient of f at the point x
    """
    n, = x.shape
    gradient = np.zeros(n).astype('float64')

    for i in range(n):
        # e_i is the vector used to add delta to the vector x for finite differencing
        e_i = np.zeros_like(x)
        e_i[i] = 1
        gradient[i] = (f(x + delta * e_i) - f(x - delta * e_i)) / (2 * delta)

    return gradient


def jacobian(f, x, delta=0.01):
    """
    Returns the Jacobian of function f at the point x
    Parameters:
        f (numpy.array -> numpy.array): A function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (f(x).shape[0], x.shape[0])
                            which is the jacobian of f at the point x
    """
    n, = x.shape
    m, = f(x).shape
    x = x.astype('float64') #Need to ensure dtype=np.float64 and also copy input. 
    jacobian = np.zeros((m, n)).astype('float64')

    # applying gradient function to each component of the x vector and making that a row of the output matrix
    for i in range(m):
        f_i = lambda x: f(x)[i]
        jacobian[i, :] = gradient(f_i, x, delta)
    
    return jacobian



def hessian(f, x, delta=0.01):
    """
    Returns the Hessian of function f at the point x
    Parameters:
        f (numpy.array -> double): A scalar function accepts numpy array x
        x (numpy.array): A numpy array which is the same form as the argument supplied to f
        delta (double): delta used in the finite difference method

    Returns:
        ret (numpy.array): A 2D numpy array with shape (x.shape[0], x.shape[0])
                            which is the Hessian of f at the point x
    """
    # Here it is important to note that the Hessian of a function is the Jacobian of its Gradient!
    grad_f = lambda x: gradient(f, x, delta)
    hessian = jacobian(grad_f, x, delta)

    return hessian
    


