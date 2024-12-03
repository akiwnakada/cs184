import argparse
import os

import gymnasium as gym
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from tqdm import trange

from dataset import ExpertDataset
from utils import (
    ENVIRONMENTS,
    get_base_parser,
    get_dataset_path,
    get_expert_path,
    print_args,
)


def make_dataset(policy: DQN, env: gym.Env, args):
    """Makes an expert dataset from `args.num_trajectories` trajectories from `policy`."""
    all_states, all_actions = [], []
    for _ in trange(args.num_trajectories, desc="Expert trajectories"):
        t_states, t_actions = [], []
        state, _info = env.reset()
        term, trunc = False, False

        while not (term or trunc):
            action, _ = policy.predict(state, deterministic=True)
            next_state, _r, term, trunc, _info = env.step(action)

            t_states.append(torch.from_numpy(state))
            t_actions.append(torch.tensor(action))

            state = next_state

        all_states += t_states
        all_actions += t_actions

    state_tensor = torch.stack(all_states)
    action_tensor = torch.stack(all_actions)

    dataset = ExpertDataset(state_tensor, action_tensor)

    os.makedirs(args.data_dir, exist_ok=True)
    dataset_path = get_dataset_path(args)
    torch.save(dataset, dataset_path)
    print(f"Expert dataset saved to {dataset_path}")


def collect_expert_data(args):
    env = gym.make(args.env)
    policy_path = get_expert_path(args)

    if args.learn:
        policy_args, timesteps = (
            (mountain_car_args, mountain_car_timesteps)
            if args.env == "MountainCar-v0"
            else (cart_pole_args, cart_pole_timesteps)
        )
        policy = DQN(
            **policy_args,
            env=env,
            verbose=1,
            seed=args.seed,
        )
        policy.learn(total_timesteps=timesteps)

        # save the policy
        os.makedirs(args.expert_save_path, exist_ok=True)
        policy.save(policy_path)
        del policy

    # reload policy and check
    policy = DQN.load(policy_path, env)
    mean_return, std_return = evaluate_policy(
        policy, policy.get_env(), n_eval_episodes=20
    )
    print(f"mean reward: {mean_return:.2f} +/- {std_return:.2f}")
    make_dataset(policy, env, args)
    env.close()

    # view a trajectory from the policy
    visualize_policy(policy, args)


def get_args():
    """Parses command line arguments."""
    parser = get_base_parser(
        "Collect expert data for imitation learning", expert_only=True
    )
    parser.add_argument(
        "--learn",
        action="store_true",
        help="learn an expert policy from scratch",
    )
    parser.add_argument(
        "--num_trajectories",
        type=int,
        default=50,
        help="number of trajectories to collect",
    )
    args = parser.parse_args()
    print_args(args)
    return args


def visualize_policy(policy: DQN, args):
    env = gym.make(args.env, render_mode="human")
    state, _info = env.reset()
    term, trunc = False, False
    while not (term or trunc):
        env.render()
        action, _state = policy.predict(state, deterministic=True)
        state, _r, term, trunc, _info = env.step(action)
    env.close()


# see hyperparameters from https://huggingface.co/sb3/dqn-CartPole-v1
cart_pole_args = {
    "batch_size": 64,
    "buffer_size": 100000,
    "exploration_final_eps": 0.04,
    "exploration_fraction": 0.16,
    "gamma": 0.99,
    "gradient_steps": 128,
    "learning_rate": 0.0023,
    "learning_starts": 1000,
    "policy": "MlpPolicy",
    "policy_kwargs": dict(net_arch=[256, 256]),
    "target_update_interval": 10,
    "train_freq": 256,
}
cart_pole_timesteps = 50000

# see hyperparameters from https://huggingface.co/sb3/dqn-MountainCar-v0
mountain_car_args = {
    "batch_size": 128,
    "buffer_size": 10000,
    "exploration_final_eps": 0.07,
    "exploration_fraction": 0.2,
    "gamma": 0.98,
    "gradient_steps": 8,
    "learning_rate": 0.004,
    "learning_starts": 1000,
    "policy": "MlpPolicy",
    "policy_kwargs": dict(net_arch=[256, 256]),
    "target_update_interval": 600,
    "train_freq": 16,
}
mountain_car_timesteps = 120000


if __name__ == "__main__":
    # generate a dataset from an expert policy
    args = get_args()
    collect_expert_data(args)
