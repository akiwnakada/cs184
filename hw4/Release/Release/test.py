import gymnasium as gym
import numpy as np
from tqdm import trange

from learner import BC, DAgger, ImitationLearner
from utils import get_base_parser, get_policy_path


def collect_trajectory(env: gym.Env, model: ImitationLearner):
    term, trunc = False, False
    state, _info = env.reset()

    length = 0
    reward = 0

    # collect a trajectory by choosing the most likely action
    while not (term or trunc):
        logits = model.get_logits(state)
        a = logits.argmax().item()

        next_state, r, term, trunc, _info = env.step(a)
        state = next_state
        length += 1
        reward += r

    return length, reward


def test(args):
    """Visualize the learned policy for `num_trajectories` episodes."""
    env = gym.make(args.env, render_mode="human" if args.render else None)
    state_dim = env.observation_space.shape[0]
    if isinstance(env.action_space, gym.spaces.Discrete):
        action_dim = env.action_space.n
    else:
        action_dim = env.action_space.shape[0]

    algo = DAgger if args.algo == "dagger" else BC
    model = algo(state_dim, action_dim, args)
    model.load(get_policy_path(args))
    lengths, rewards = [], []
    for _ in trange(args.num_trajectories, desc="Evaluation"):
        length, reward = collect_trajectory(env, model)
        lengths.append(length)
        rewards.append(reward)

    print(
        f"average episode length across {args.num_trajectories} trajectories: {np.mean(lengths):.2f} ± {np.std(lengths):.2f}"
    )
    print(
        f"average total reward across {args.num_trajectories} trajectories: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}"
    )


def get_test_args():
    parser = get_base_parser("Evaluate a learned policy")
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument(
        "--num_trajectories", type=int, default=10, help="number of trajectories"
    )
    args = parser.parse_args()
    args.lr = 1e-3  # dummy value
    return args


if __name__ == "__main__":
    args = get_test_args()
    test(args)
