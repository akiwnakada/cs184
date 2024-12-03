from sklearn.kernel_approximation import RBFSampler
import numpy as np

rbf_feature = RBFSampler(gamma=1, random_state=12345)


def extract_features(state, num_actions):
    """ This function computes the RFF features for a state for all the discrete actions

    :param state: column vector of the state we want to compute phi(s,a) of (shape |S|x1)
    :param num_actions: number of discrete actions you want to compute the RFF features for
    :return: phi(s,a) for all the actions (shape 100x|num_actions|)
    """
    s = state.reshape(1, -1)
    s = np.repeat(s, num_actions, 0)
    a = np.arange(0, num_actions).reshape(-1, 1)
    sa = np.concatenate([s,a], -1)
    feats = rbf_feature.fit_transform(sa)
    feats = feats.T
    return feats


def compute_softmax(logits, axis):
    """ computes the softmax of the logits

    :param logits: the vector to compute the softmax over
    :param axis: the axis we are summing over
    :return: the softmax of the vector

    Hint: to make the softmax more stable, subtract the max from the vector before applying softmax
    """
    max = np.max(logits, axis = axis)
    stable_vec = np.exp(logits - max)
    return (stable_vec/np.sum(stable_vec, axis = axis))

def compute_action_distribution(theta, phis):
    """ compute probability distrubtion over actions

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :return: softmax probability distribution over actions (shape 1 x |A|)
    """
    logits = np.dot(theta.T, phis)
    return compute_softmax(logits, axis = 1)


def compute_log_softmax_grad(theta, phis, action_idx):
    """ computes the log softmax gradient for the action with index action_idx

    :param theta: model parameter (shape d x 1)
    :param phis: RFF features of the state and actions (shape d x |A|)
    :param action_idx: The index of the action you want to compute the gradient of theta with respect to
    :return: log softmax gradient (shape d x 1)
    """
    phi = phis[:, action_idx].reshape(-1,1)
    probs = compute_action_distribution(theta, phis)
    expectation = np.dot(phis, probs.T)
    res = phi - expectation
    return res


def compute_fisher_matrix(grads, lamb=1e-3):
    """ computes the fisher information matrix using the sampled trajectories gradients

    :param grads: list of list of gradients, where each sublist represents a trajectory (each gradient has shape d x 1)
    :param lamb: lambda value used for regularization 

    :return: fisher information matrix (shape d x d)

    Note: don't forget to take into account that trajectories might have different lengths
    """
    d = grads[0][0].shape[0]
    fisher_information_matrix = np.zeros((d,d))
    I = np.eye(d)
    N = len(grads)

    for trajectory in grads:
        H = len(trajectory)
        for grad in trajectory:
            fisher_information_matrix += np.dot(grad, grad.T)/H
    
    fisher_information_matrix /= N
    fisher_information_matrix += lamb * I

    return fisher_information_matrix


def compute_value_gradient(grads, rewards):
    """ computes the value function gradient with respect to the sampled gradients and rewards

    :param grads: ist of list of gradients, where each sublist represents a trajectory
    :param rewards: list of list of rewards, where each sublist represents a trajectory
    :return: value function gradient with respect to theta (shape d x 1)

    I assume that here we are computing the gradient of the value function here using the reinforce policy algorithm, the only issue is that we don't have access
    to the advantage function since we don't have access to the value and q functions. Instead, I guess we could just use the sum of rewards? 
    """
    d = grads[0][0].shape[0]
    policy_grad = np.zeros((d,1))
    N = len(grads)
    baseline = sum(sum(trajectory) for trajectory in rewards)/N
    for trajectory_rewards, trajectory_grads in zip(rewards, grads):
        H = len(trajectory_rewards)
        for time_step in range(len(trajectory_grads)):
            policy_grad += trajectory_grads[time_step] * (sum(trajectory_rewards[time_step:]) - baseline)/H
    policy_grad /= N

    return policy_grad


def compute_eta(delta, fisher, v_grad):
    """ computes the learning rate for gradient descent

    :param delta: trust region size
    :param fisher: fisher information matrix (shape d x d)
    :param v_grad: value function gradient with respect to theta (shape d x 1)
    :return: the maximum learning rate that respects the trust region size delta
    """
    d = fisher.shape[0]
    epsilon = 10 ** (-6)
    I = np.eye(d)
    fisher_inv = np.linalg.solve(fisher, I)
    denominator = v_grad.T @ fisher_inv @ v_grad + epsilon

    eta = (delta/denominator)**(0.5)

    return eta

