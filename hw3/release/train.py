import gym
import numpy as np
from utils import *
import matplotlib.pyplot as plt


def sample(theta, env, N):
    """ samples N trajectories using the current policy

    :param theta: the model parameters (shape d x 1)
    :param env: the environment used to sample from
    :param N: number of trajectories to sample
    :return:
        trajectories_gradients: lists with sublists for the gradients for each trajectory rollout (should be a 2-D list)
        trajectories_rewards:  lists with sublists for the rewards for each trajectory rollout (should be a 2-D list)

    Note: the maximum trajectory length is 200 steps
    """
    total_rewards = []
    total_grads = []

    for _ in range(N):
        done = False
        state = env.reset()
        
        trajectory_rewards = []
        trajectory_grads = []

        while not done:
            phis = extract_features(state, env.action_space.n)
            action = np.random.choice(env.action_space.n, p=compute_action_distribution(theta, phis).flatten())
            trajectory_grads.append(compute_log_softmax_grad(theta, phis, action))
            state, reward, done, _ = env.step(action)
            trajectory_rewards.append(reward)
        total_rewards.append(trajectory_rewards)
        total_grads.append(trajectory_grads)

    return total_grads, total_rewards


def train(N, T, delta, lamb=1e-3):
    """

    :param N: number of trajectories to sample in each time step
    :param T: number of iterations to train the model
    :param delta: trust region size
    :param lamb: lambda for fisher matrix computation
    :return:
        theta: the trained model parameters
        avg_episodes_rewards: list of average rewards for each time step
    """
    theta = np.random.rand(100,1)
    env = gym.make('CartPole-v0')
    env.seed(12345)

    episode_rewards = []
    for _ in range(T):
        total_grads, total_rewards = sample(theta, env, N)
        avg_rewards = sum(sum(trajectory) for trajectory in total_rewards)/N
        episode_rewards.append(avg_rewards)

        fisher = compute_fisher_matrix(total_grads, lamb)
        v_grad = compute_value_gradient(total_grads, total_rewards)
        eta = compute_eta(delta, fisher, v_grad)
        I = np.eye(fisher.shape[0])
        fisher_inv = np.linalg.solve(fisher, I)
        theta += eta * fisher_inv @ v_grad

    return theta, episode_rewards

if __name__ == '__main__':
    np.random.seed(1234)
    theta, episode_rewards = train(N=100, T=20, delta=1e-2)
    print(episode_rewards)
    plt.plot(episode_rewards)
    plt.title("avg rewards per timestep")
    plt.xlabel("timestep")
    plt.ylabel("avg rewards")
    plt.show()
