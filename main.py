from pursuit import MAWaterWorld_mod
from MADDPG import MADDPG
import numpy as np
import torch as th
import visdom
from params import scale_reward
from arguments import get_common_args
from envs.SumoAgent import SumoAgent

arg = get_common_args()

reward_record = []

# numpy的初始化种子
seed = arg.seed
np.random.seed(seed)
# torch的初始化种子
th.manual_seed(1234)
n_agents = arg.n_agents
n_states = arg.n_states
n_actions = arg.n_actions
capacity = 1000000
batch_size = arg.batch_size

n_episode = arg.n_episode
episode_length = arg.episode_length
episodes_before_train = arg.episodes_before_train

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

sumoBinary = r"D:/SUMO/bin/sumo-gui"
sumoBinary_nogui = r"D:/SUMO/bin/sumo"
sumo_path = 'D:/zbb99/Desktop/pytorch-maddpg'
sumoCmd = [sumoBinary_nogui,
            '-c',
            r'{0}/exp.sumocfg'.format(sumo_path)]

env = SumoAgent(sumoCmd, n_agents)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    obs = env.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for t in range(episode_length):
        obs = obs.type(FloatTensor)
        actions = maddpg.select_action(obs).data.cpu()
        # 将actions 转化为列表的形式
        acts_np = actions.numpy().tolist()
        obs_, reward, done, _ = env.step(acts_np)

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != episode_length - 1:
            next_obs = obs_
        else:
            next_obs = None

        total_reward += reward.sum()
        rr += reward.cpu().numpy()
        maddpg.memory.push(obs.data, actions, next_obs, reward)
        obs = next_obs

        c_loss, a_loss = maddpg.update_policy()
    maddpg.episode_done += 1
    print('Episode: %d, reward = %f' % (i_episode, total_reward))
    reward_record.append(total_reward)

    if maddpg.episode_done == maddpg.episodes_before_train:
        print('training now begins...')
        print('MADDPG on WaterWorld\n' +
              'scale_reward=%f\n' % scale_reward +
              'agent=%d' % n_agents )
env.end_sumo()
