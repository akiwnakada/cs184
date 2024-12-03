import os

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import trange

from dataset import ExpertDataset, get_dataloader
from learner import BC, DAgger
from utils import get_base_parser, get_dataset_path, get_policy_path, print_args


def train_bc(expert_dataset: ExpertDataset, learner: BC, args):
    """Repeat `args.bc_epochs` times: For each batch in the dataloader,
    run a gradient descent step with it using `learner.learn`,
    and append the average loss per state-action pair to `epoch_losses`.
    """
    epoch_losses = []
    dataloader = get_dataloader(expert_dataset, args)


    for _ in range(args.bc_epochs):
        total_loss = 0
        num_samples = 0
        for states, actions in dataloader:
            loss = learner.learn(states, actions)
            total_loss += loss
            num_samples += states.size(0)
        avg_loss = total_loss/num_samples
        epoch_losses.append(avg_loss)
    
    return epoch_losses


def train_dagger(env: gym.Env, expert_dataset: ExpertDataset, learner: DAgger, args):
    """Trains a policy using DAgger.

    Repeat `args.dagger_epochs` times:

    1. Collect a trajectory of `args.num_rollout_steps` expert state-action pairs using `learner.rollout`.
    2. Add this to the `expert_dataset` (see the definition of `ExpertDataset` in `dataset.py`).
    2. Use `train_bc` to train `learner` on the `expert_dataset` with gradient descent.

    Each element of `epoch_losses` should be the average of the supervised learning losses from the corresponding call to `train_bc`.
    """
    epoch_losses = []

    for _ in range(args.dagger_epochs):
        trajectory = learner.rollout(env, args.num_rollout_steps)
        expert_dataset.add_data(trajectory)
        losses = train_bc(expert_dataset, learner, args)
        avg_loss = sum(losses)/len(losses)
        epoch_losses.append(avg_loss)

    return epoch_losses


def experiment(args):
    """Executes an imitation learning experiment for the given configuration."""
    env = gym.make(args.env)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.dagger:
        expert_dataset = ExpertDataset(
            torch.tensor([]),
            torch.tensor([], dtype=int),
        )
        learner = DAgger(state_dim, action_dim, args)
        epoch_losses = train_dagger(env, expert_dataset, learner, args)
        num_epochs = args.dagger_epochs
    else:
        save_path = get_dataset_path(args)
        expert_dataset = torch.load(save_path)
        learner = BC(state_dim, action_dim, args)
        epoch_losses = train_bc(expert_dataset, learner, args)
        num_epochs = args.bc_epochs

    # plotting
    epochs = np.arange(1, num_epochs + 1)
    plot_losses(epochs, epoch_losses, args)

    # saving policy
    os.makedirs(args.policy_save_dir, exist_ok=True)
    policy_save_path = get_policy_path(args)
    learner.save(policy_save_path)
    print("Saved learned policy to", policy_save_path)


def get_args():
    """Parses command line arguments."""
    parser = get_base_parser(description="Imitation learning")

    parser.add_argument(
        "--plots_dir",
        default="./plots",
        help="directory to save plots to",
    )

    # behavioral cloning args
    parser.add_argument(
        "--bc_epochs", type=int, help="number of supervised learning epochs"
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="learning rate",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="batch size",
    )
    parser.add_argument(
        "--num_dataset_samples",
        type=int,
        help="number of samples to start dataset off with",
    )

    # DAgger args
    parser.add_argument(
        "--num_rollout_steps",
        type=int,
        help="number of steps to roll out with the policy",
    )
    parser.add_argument(
        "--dagger_epochs", type=int, help="number of steps to run dagger"
    )

    parser.set_defaults(**get_default_args() or {})

    args = parser.parse_args()
    print_args(args)
    args.dagger = args.algo == "dagger"
    return args


def get_default_args() -> dict:
    pass  # YOUR SOLUTION HERE


# Plotting utils
def plot_losses(
    epochs,
    losses,
    args,
):
    plt.plot(epochs, losses)
    plt.title(f"{args.algo} losses for {args.env}")
    plt.xlabel("epochs")
    plt.ylabel("loss")

    os.makedirs(args.plots_dir, exist_ok=True)
    fig_path = os.path.join(args.plots_dir, f"{args.algo}_{args.env}.png")
    plt.savefig(fig_path)
    print(f"Saved loss plot to {fig_path}")
    plt.show()


if __name__ == "__main__":
    args = get_args()
    experiment(args)
