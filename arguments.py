import argparse

"""
Here are the param for the training

"""

def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--n_agents', type=int, default=9, help='number of agents')
    parser.add_argument('--n_states', type=int, default=2, help='number of states')
    parser.add_argument('--n_actions', type=int, default=1, help='number of actions')
    parser.add_argument('--capacity', type=int, default=100000, help='total capacity of ReplayMemory')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for model training')
    parser.add_argument('--n_episode', type=int, default=100, help='number of episodes to train')
    parser.add_argument('--episodes_before_train', type=int, default=10000, help='number of episodes to warm up before train')
    parser.add_argument('--episode_length', type=int, default=1000, help='the step length of one episode')
    parser.add_argument('--steps_per_update', type=int, default=100, help='the step_length to update')
    # scale_reward maybe not useful
    parser.add_argument('--scale_reward', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--manual_seed', type=int, default=1234, help='Radndom seed of torch')
    parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dim of model')
    parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--tau', type=float, default=0.01, help='Target update rate')
    parser.add_argument('--min_eps', type=float, default=0.36786, help='min_eps, if eps < min_eps, eps=0, to exploit instead of exploring, 10000')
    parser.add_argument('--eps', type=float, default=1, help='eps, to decide explore or exploit')
    parser.add_argument('--gamma', type=float, default=0.9999, help='the discount of eps')

    args = parser.parse_args()
    return args