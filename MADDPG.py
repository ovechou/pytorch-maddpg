from model import Critic, Actor
import torch as th
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
from randomProcess import OrnsteinUhlenbeckProcess
import torch.nn as nn
import numpy as np
from params import scale_reward


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)


class MADDPG:
    def __init__(self, n_agents, dim_obs, dim_act, batch_size,
                 capacity, episodes_before_train):
        self.actors = [Actor(dim_obs, dim_act) for i in range(n_agents)]
        self.critics = [Critic(n_agents, dim_obs,
                               dim_act) for i in range(n_agents)]
        # target actors和target critics使用deepcopy 被复制对象完全再复制一遍作为独立的新个体单独存在。所以改变原有被复制对象不会对已经复制出来的新对象产生影响。
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)

        self.n_agents = n_agents
        self.n_states = dim_obs
        self.n_actions = dim_act
        self.memory = ReplayMemory(capacity)
        self.batch_size = batch_size
        self.use_cuda = th.cuda.is_available()
        self.episodes_before_train = episodes_before_train

        self.GAMMA = 0.95
        self.tau = 0.01

        self.var = [1.0 for i in range(n_agents)]
        self.critic_optimizer = [Adam(x.parameters(),
                                      lr=0.001) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),
                                     lr=0.0001) for x in self.actors]

        # 使用cuda加速
        if self.use_cuda:
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

        self.steps_done = 0
        # episode_done用来计算已经跑过了多少回合，来确定是否结束预训练
        self.episode_done = 0

    def update_policy(self):
        # do not train until exploration is enough
        if self.episode_done <= self.episodes_before_train:
            return None, None

        ByteTensor = th.cuda.ByteTensor if self.use_cuda else th.ByteTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor

        # actor Loss
        c_loss = []
        # critic Loss
        a_loss = []
        # 循环，对于每一个agent提取transitions
        # TODO:后面继续看没有智能体的区分吗？
        # 因为智能体统一，可以通用来训练
        for agent in range(self.n_agents):
            # 提取过渡态
            transitions = self.memory.sample(self.batch_size)
            # 利用*号操作符，可以将元组解压为列表. *transitions将transtions解压为列表
            # zip(*transitions) 得到的结果是[(state1, state2), (action1, action2), (next_state1, next_state2), (reward1, reward2)] 
            # batch = Experience(states=(1, 5), actions=(2, 6), next_states=(3, 7), rewards=(4, 8))
            batch = Experience(*zip(*transitions))
            # 是否终止状态
            # list(map(...))返回的数值: [True, True]
            # ByteTensor后返回的数值tensor([1, 1], dtype=torch.uint8)
            non_final_mask = ByteTensor(list(map(lambda s: s is not None,
                                                 batch.next_states)))
            # state_batch: batch_size x n_agents x dim_obs
            # state_batch的格式: tensor([1., 1.])
            state_batch = th.stack(batch.states).type(FloatTensor)
            action_batch = th.stack(batch.actions).type(FloatTensor)
            reward_batch = th.stack(batch.rewards).type(FloatTensor)
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor)

            # for current agent
            # 使用view重新塑形
            # whole_state的格式为([batch_size, n_agents ✖ dim_obs])
            whole_state = state_batch.view(self.batch_size, -1)
            whole_action = action_batch.view(self.batch_size, -1)

            # 把critic优化器梯度置零，也就是把loss关于weight的导数变成0
            self.critic_optimizer[agent].zero_grad()
            # 当前的Q值, 使用当前critic来进行评估
            current_Q = self.critics[agent](whole_state, whole_action)
            
            non_final_next_actions = [
                self.actors_target[i](non_final_next_states[:,
                                                            i,
                                                            :]) for i in range(
                                                                self.n_agents)]
            non_final_next_actions = th.stack(non_final_next_actions)
            # transpose: 交换维度0和1，即转置
            # contiguous操作保证张量是连续的，方便后续的view操作
            non_final_next_actions = (
                non_final_next_actions.transpose(0,
                                                 1).contiguous())

            # TODO: 对此处代码不深究，涉及到数学内容，直接套用
            # target_Q初始化
            target_Q = th.zeros(
                self.batch_size).type(FloatTensor)
            

            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, self.n_agents * self.n_states),
                non_final_next_actions.view(-1,
                                            self.n_agents * self.n_actions)
            ).squeeze()
            # scale_reward: to scale reward in Q functions

            target_Q = (target_Q.unsqueeze(1) * self.GAMMA) + (
                reward_batch[:, agent].unsqueeze(1) * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = state_batch[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = action_batch.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(self.batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)

        if self.steps_done % 100 == 0 and self.steps_done > 0:
            for i in range(self.n_agents):
                soft_update(self.critics_target[i], self.critics[i], self.tau)
                soft_update(self.actors_target[i], self.actors[i], self.tau)

        return c_loss, a_loss

    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = th.zeros(
            self.n_agents,
            self.n_actions)
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()

            act += th.from_numpy(
                np.random.randn(2) * self.var[i]).type(FloatTensor)

            if self.episode_done > self.episodes_before_train and\
               self.var[i] > 0.05:
                self.var[i] *= 0.999998
            act = th.clamp(act, -1.0, 1.0)

            actions[i, :] = act
        self.steps_done += 1

        return actions
