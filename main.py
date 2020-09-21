from MADDPG import MADDPG
import numpy as np
import torch as th
# import visdom
from params import scale_reward
from arguments import get_common_args
from envs.SumoAgent import SumoAgent
from pathlib import Path
import os
import time

args = get_common_args()

reward_record = []

# numpy的初始化种子
seed = args.seed
np.random.seed(seed)
# torch的初始化种子
manual_seed = args.manual_seed
th.manual_seed(manual_seed)
n_agents = args.n_agents
n_states = args.n_states
n_actions = args.n_actions
capacity = 100000
batch_size = args.batch_size


n_episode = args.n_episode
episode_length = args.episode_length
episodes_before_train = args.episodes_before_train

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

sumoBinary = r"/usr/share/sumo/bin/sumo-gui"
sumoBinary_nogui = r"/usr/share/sumo/bin/sumo"

sumo_path = '/userhome/ove/pytorch-maddpg'
sumoCmd = [sumoBinary_nogui,
            '-c',
            r'{0}/exp.sumocfg'.format(sumo_path)]

def run(args):
    # 创建模型文件夹
    time_now = time.strftime('%y_%m_%d_%H_%M_%S', time.localtime())
    
    env = SumoAgent(sumoCmd, n_agents)

    FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
    eps = args.eps
    gamma = args.gamma
    min_eps = args.min_eps
    for i_episode in range(n_episode):
        total_reward = 0.0
        rr = np.zeros((n_agents,))
        for t in range(episode_length):
            eps = eps * gamma
            if eps < min_eps:
                eps = 0.0
            obs = env.get_obs()
            # 将obs转化为tensor格式的
            torch_obs = th.tensor(obs)
            torch_obs = torch_obs.type(FloatTensor)
            actions = maddpg.select_action(torch_obs, eps, min_eps).data.cpu()
            # 将actions转化为torch.tensor(9,1)的格式
            torch_acs = th.LongTensor(9,1)
            for i in range(n_agents):
                torch_acs[i] = actions[i].unsqueeze(0)
            # 将actions 转化为列表的形式
            acts_np = actions.numpy().tolist()
            reward = env.step(acts_np, time_now)
            obs_ = env.get_obs()
            reward = th.FloatTensor(reward).type(FloatTensor)
            obs_ = np.stack(obs_)
            obs_ = th.from_numpy(obs_).float()
            if t != episode_length - 1:
                next_obs = obs_
            else:
                next_obs = None
            total_reward += reward.sum()
            # rr += reward.cpu().numpy()
            maddpg.memory.push(torch_obs, torch_acs, next_obs, reward)
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

if __name__ == '__main__':
    args = get_common_args()
    run(args)


# if maddpg.episode_done >= 4:
#     test_memory = maddpg.memory.sample(3)
#     batchtest = Experience(*zip(*test_memory))
#     state_batch = Variable(th.stack(batchtest.states).type(FloatTensor))
#     whole_state = state_batch.view(3, -1)
#     print(test_memory)
#     print(batchtest)