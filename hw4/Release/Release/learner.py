import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from jaxtyping import Float, Int
from stable_baselines3 import DQN
from torch import Tensor

from dataset import ExpertDataset
from utils import DiscretePolicy, get_expert_path

"""Imitation learning agents file."""


class ImitationLearner:
    def __init__(self, state_dim: int, action_dim: int, args):
        # Policy network setup
        self.policy = DiscretePolicy(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.loss = nn.CrossEntropyLoss(reduction="sum")

    def load(self, path: str):
        self.policy.load_state_dict(torch.load(path))

    def save(self, path: str):
        torch.save(self.policy.state_dict(), path)

    def get_logits(
        self, states: Float[Tensor, "B state_dim"]
    ) -> Float[Tensor, "B action_dim"]:
        """Returns the action distribution for each state in the batch."""
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        return self.policy(states)

    def learn(
        self,
        expert_states: Float[Tensor, "B state_dim"],
        expert_actions: Int[Tensor, "B"],
    ):
        """Takes in a batch of expert state-action pairs and performs a step of gradient descent:
        1. Compute the current policy's action distribution at each state.
        2. Calculate the cross-entropy loss between each resulting action distribution
           and the corresponding expert action (thought of as a distribution that takes on a single value).
        3. Update the policy parameters by gradient descent on the loss.

        Returns the total cross-entropy loss over this batch.
        """
        expert_actions = expert_actions.squeeze()
        self.optimizer.zero_grad()
        actions_distributions = self.get_logits(expert_states)
        loss = self.loss(actions_distributions, expert_actions)
        loss.backward()
        self.optimizer.step()

        return loss.item()


class BC(ImitationLearner):
    pass  # BC is identical to ImitationLearner


class DAgger(ImitationLearner):
    """Implements the Dataset Aggregation algorithm for imitation learning."""

    def __init__(self, state_dim: int, action_dim: int, args):
        super().__init__(state_dim, action_dim, args)
        # get access to the expert policy
        self.expert_policy = DQN.load(get_expert_path(args))

    def rollout(self, env: gym.Env, num_steps: int):
        """Obtain expert actions over a trajectory obtained by the current policy.

        1. Take actions in the environment according to `self.policy` to obtain a trajectory of length `num_steps`.
        2. Return an `ExpertDataset` object with the visited states and the corresponding actions from `self.expert_policy`.

        See the definition of `ExpertDataset` in `dataset.py`.
        """
        states = []
        expert_actions = []
        state, _ = env.reset()
        done = False
        counter = 0

        while not done and counter < num_steps:
            states.append(state)

            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            logits = self.get_logits(state_tensor)

            action = self.sample_from_logits(logits.squeeze(0)).item()

            expert_action = self.expert_policy.predict(np.array([state]))[0]
            expert_actions.append(expert_action)

            next_state, _, done, _, _ = env.step(action)

            if done:
                state, _ = env.reset()
            else:
                state = next_state
            counter += 1

        states = torch.FloatTensor(np.array(states))
        expert_actions = torch.LongTensor(np.array(expert_actions))

        return ExpertDataset(states, expert_actions)
    

    def sample_from_logits(self, logits: Float[Tensor, "action_dim"]):
        """Takes in a distribution over actions, specified by `logits`, and samples an action from this distribution."""
        num_samples = 1
        probs = torch.softmax(logits, dim = -1)
        sample = torch.multinomial(probs, num_samples)

        return sample
