import argparse

"""
Here are the param for the training

"""

def get_common_args():
    parser = argparse.ArgumentParser()
    # the environment setting
    parser.add_argument('--n_agents', type=int, default=9, help='number of agents')
    parser.add_argument('--n_states', type=int, default=2, help='number of states')
    parser.add_argument('--n_actions', type=int, default=2, help='number of actions')
    parser.add_argument('--capacity', type=int, default=100000, help='total capacity of ReplayMemory')
    parser.add_argument('--batch_size', type=int, default=1000, help='the size of one batch to train')
    parser.add_argument('--n_episodes', type=int, default=20000, help='number of episodes to train')
    parser.add_argument('--max_steps', type=int, default=1000)
    parser.add_argument('--episode_before_train', type=int, default=100, help='number of episodes to warm up before train')

    