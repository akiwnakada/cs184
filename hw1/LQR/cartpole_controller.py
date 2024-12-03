import numpy as np
from finite_difference_method import gradient, jacobian, hessian
from lqr import lqr

class LocalLinearizationController:
    def __init__(self, env):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset 
                 the state to any state
        """
        self.env = env

    def c(self, x, u):
        """
        Cost function of the env.
        It sets the state of environment to `x` and then execute the action `u`, and
        then return the cost. 
        Parameter:
            x (1D numpy array) with shape (4,) 
            u (1D numpy array) with shape (1,)
        Returns:
            cost (double)
        """
        assert x.shape == (4,)
        assert u.shape == (1,)
        env = self.env
        env.reset(state=x)
        observation, cost, done, info = env.step(u)
        return cost

    def f(self, x, u):
        """
        State transition function of the environment.
        Return the next state by executing action `u` at the state `x`
        Parameter:
            x (1D numpy array) with shape (4,)
            u (1D numpy array) with shape (1,)
        Returns:
            next_observation (1D numpy array) with shape (4,)
        """
        assert x.shape == (4,)
        assert u.shape == (1,)
        env = self.env
        env.reset(state=x)
        next_observation, cost, done, info = env.step(u)
        return next_observation

    # New function we made to ensure the approximate cost function is convex
    def convexify(self, Q, M, R):
        n_x = Q.shape[0]
        n_u = R.shape[0]  

        upper = np.hstack((Q, M))
        lower = np.hstack((M.T, R))
        W = np.vstack((upper, lower))

        eigenvalues, eigenvectors = np.linalg.eigh(W)

        # Only apply function if the matrix W isn't positive definite already
        if np.all(eigenvalues > 0):
            return Q, M, R 

        lambda_value = 1e-6
        eigenvalues_modified = np.maximum(eigenvalues, 0) + lambda_value  
        W_positive_definite = eigenvectors @ np.diag(eigenvalues_modified) @ eigenvectors.T

        Q_updated = W_positive_definite[:n_x, :n_x]
        M_updated = W_positive_definite[:n_x, n_x:n_x + n_u]
        R_updated = W_positive_definite[n_x:n_x + n_u, n_x:n_x + n_u]

        return Q_updated, M_updated, R_updated
    

    def compute_local_policy(self, x_star, u_star, T):
        """
        This function perform a first order taylar expansion function f and
        second order taylor expansion of cost function around (x_star, u_star). Then
        compute the optimal polices using lqr.
        outputs:
        Parameters:
            T (int) maximum number of steps
            x_star (numpy array) with shape (4,)
            u_star (numpy array) with shape (1,)
        return
            Ks(List of tuples (K_i,k_i)): A list [(K_0,k_0), (K_1, k_1),...,(K_T,k_T)] with length T
                                          Each K_i is 2D numpy array with shape (1,4) and k_i is 1D numpy
                                          array with shape (1,)
                                          such that the optimial policies at time are i is K_i * x_i + k_i
                                          where x_i is the state
        """
        # For this problem I basically just have to assign the different derivatives to the different inputs to the LQR function then run that
        # The formulas for each of the different inputs to the LQR function are derived in lecture

        delta = 0.01 

        A = jacobian(lambda x: self.f(x, u_star), x_star, delta)
        B = jacobian(lambda u: self.f(x_star, u), u_star, delta)
        Q = hessian(lambda x: self.c(x, u_star), x_star, delta)
        R = hessian(lambda u: self.c(x_star, u), u_star, delta)
        q = gradient(lambda x: self.c(x, u_star), x_star, delta)
        r = gradient(lambda u: self.c(x_star, u), u_star, delta)
        M = jacobian(lambda x: gradient(lambda u: self.c(x, u), u_star, delta), x_star, delta).T
        b = np.array([self.c(x_star, u_star)]) 
        
        Q, M, R = self.convexify(Q, M, R)
        m = self.f(x_star, u_star) - A @ x_star - B @ u_star

        # We use flatten function again to ensure the correct shape of the output
        Ks = lqr(A, B, m[:, np.newaxis], Q, R, M, q[:, np.newaxis], r[:, np.newaxis], b, T)
        return [(K, k.flatten()) for K, k in Ks]


class PIDController:
    """
    Parameters:
        P, I, D: Controller gains
    """

    def __init__(self, P, I, D):
        """
        Parameters:
            env: an customized openai gym environment with reset function to reset
                 the state to any state
        """
        self.P, self.I, self.D = P, I, D
        self.err_sum = 0.
        self.err_prev = 0.

    def get_action(self, err):
        self.err_sum += err
        a = self.P * err + self.I * self.err_sum + self.D * (err - self.err_prev)
        self.err_prev = err
        return a
