import torch
from jaxtyping import Float, Int
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

class ExpertDataset(Dataset):
    """A PyTorch dataset containing a batch of expert state-action pairs.

    Contains a batch of state-action pairs (s, π(s)) where π is the expert policy.
    """

    def __init__(self, states: Float[Tensor, "B state_dim"], actions: Int[Tensor, "B"]):
        self.states = states
        self.actions = actions

    def __len__(self):
        return self.states.shape[0]

    def __getitem__(self, idx):
        state = self.states[idx]
        action = self.actions[idx]
        return state, action

    def sample(self, n: int):
        """Draw n samples from the dataset."""
        # You may be interested in customizing this!
        # idx = torch.randperm(len(self))[:n]
        idx = slice(-n, None)
        return self[idx]

    def add_data(self, data: "ExpertDataset"):
        self.states = torch.cat([self.states, data.states], dim=0)
        self.actions = torch.cat([self.actions, data.actions], dim=0)


def get_dataloader(dataset: ExpertDataset, args):
    """Generate a PyTorch dataloader of `args.num_dataset_samples` from the dataset."""
    small_states, small_actions = dataset.sample(args.num_dataset_samples)
    small_dset = ExpertDataset(small_states, small_actions)
    return DataLoader(small_dset, batch_size=args.batch_size, shuffle=True)
