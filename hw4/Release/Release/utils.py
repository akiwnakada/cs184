import argparse
import os

from jaxtyping import Float
from torch import Tensor, nn

ENVIRONMENTS = ["CartPole-v1", "MountainCar-v0"]


class DiscretePolicy(nn.Module):
    """A feedforward neural network for discrete action spaces."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim),
        )

    def forward(
        self, states: Float[Tensor, "B state_dim"]
    ) -> Float[Tensor, "B action_dim"]:
        """Returns the action distribution for each state in the batch."""
        logits = self.net(states)
        return logits.float()


def get_base_parser(description: str, expert_only=False):
    parser = argparse.ArgumentParser(description=description)

    # experiment
    parser.add_argument(
        "--env", choices=ENVIRONMENTS, required=True, help="environment"
    )

    if not expert_only:
        parser.add_argument(
            "--algo", choices=["bc", "dagger"], required=True, help="algorithm"
        )
        parser.add_argument(
            "--policy_save_dir",
            default="./learned_policies",
            help="directory containing learned policies",
        )

    parser.add_argument("--seed", type=int, default=184, help="random seed")

    # dataset directory
    parser.add_argument(
        "--data_dir", default="./expert_data", help="directory containing expert data"
    )

    # DAgger args
    parser.add_argument(
        "--expert_save_path",
        default="./expert_policies",
        help="directory containing expert policies",
    )

    return parser


def print_args(args):
    """Prints the command line arguments."""
    print("=" * 30)
    print("Running experiment with args:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    print("=" * 30)


def get_expert_path(args):
    return os.path.join(args.expert_save_path, f"{args.env}_policy.pt")


def get_dataset_path(args):
    return f"{args.data_dir}/{args.env}_dataset.pt"


def get_policy_path(args):
    return os.path.join(args.policy_save_dir, f"{args.algo}_{args.env}.pt")
