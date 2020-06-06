from pursuit import MAWaterWorld_mod
from MADDPG import MADDPG
import numpy as np
import torch as th
import visdom
from params import scale_reward

# do not render the scene
e_render = False

food_reward = 10.
poison_reward = -1.
encounter_reward = 0.01
n_coop = 2
world = MAWaterWorld_mod(n_pursuers=2, n_evaders=50,
                         n_poison=50, obstacle_radius=0.04,
                         food_reward=food_reward,
                         poison_reward=poison_reward,
                         encounter_reward=encounter_reward,
                         n_coop=n_coop,
                         sensor_range=0.2, obstacle_loc=None, )

vis = visdom.Visdom(port=5274)
reward_record = []

# numpy的初始化种子
np.random.seed(1234)
# torch的初始化种子
th.manual_seed(1234)
world.seed(1234)
n_agents = world.n_pursuers
n_states = 213
n_actions = 2
capacity = 1000000
batch_size = 1000

n_episode = 20000
max_steps = 1000
episodes_before_train = 100

win = None
param = None

maddpg = MADDPG(n_agents, n_states, n_actions, batch_size, capacity,
                episodes_before_train)

FloatTensor = th.cuda.FloatTensor if maddpg.use_cuda else th.FloatTensor
for i_episode in range(n_episode):
    # obs是从环境中获取的queue_length和waiting_time
    # obs = env.get_obs() 返回list,List中是数组
    # obs = np.asarray(obs)
    # if isinstance(obs[i], np.npadday):
        # obs[i] = th.from_numpy(obs[i]).float()
        # obs[i] = Variable(obs[i]).type(FloatTensor)
    obs = world.reset()
    obs = np.stack(obs)
    if isinstance(obs, np.ndarray):
        obs = th.from_numpy(obs).float()
    total_reward = 0.0
    rr = np.zeros((n_agents,))
    for t in range(max_steps):
        obs = obs.type(FloatTensor)
        actions = maddpg.select_action(obs).data.cpu()
        # 将actions 转化为列表的形式
        acts_np = actions.numpy().tolist()
        obs_, reward, done, _ = world.step(acts_np)

        reward = th.FloatTensor(reward).type(FloatTensor)
        obs_ = np.stack(obs_)
        obs_ = th.from_numpy(obs_).float()
        if t != max_steps - 1:
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
              'agent=%d' % n_agents +
              ', coop=%d' % n_coop +
              ' \nlr=0.001, 0.0001, sensor_range=0.3\n' +
              'food=%f, poison=%f, encounter=%f' % (
                  food_reward,
                  poison_reward,
                  encounter_reward))

world.close()
