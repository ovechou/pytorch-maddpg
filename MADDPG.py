from model import Critic, Actor
import torch as th
import torch.nn.functional as F
from copy import deepcopy
from memory import ReplayMemory, Experience
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from params import scale_reward
from torch.autograd import Variable
from torch.distributions import Categorical
import os



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

        # self.var = [1.0 for i in range(n_agents)]
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
            state_batch = Variable(th.stack(batch.states).type(FloatTensor))
            action_batch = Variable(th.stack(batch.actions).type(FloatTensor))
            reward_batch = Variable(th.stack(batch.rewards).type(FloatTensor))
            # : (batch_size_non_final) x n_agents x dim_obs
            non_final_next_states = Variable(th.stack(
                [s for s in batch.next_states
                 if s is not None]).type(FloatTensor))

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

    def select_action(self, state_batch, eps=None, min_eps=None):
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            policy = self.actors[i](sb.unsqueeze(0)).squeeze()


        # state_batch: n_agents x state_dim
        argmax_acs = th.LongTensor(
            self.n_agents,
            1)
        rand_acs = th.LongTensor(
            self.n_agents,
            1)
        LongTensor = th.cuda.LongTensor if self.use_cuda else th.LongTensor
        FloatTensor = th.cuda.FloatTensor if self.use_cuda else th.FloatTensor
        for i in range(self.n_agents):
            sb = state_batch[i, :].detach()
            actor = self.actors[i]
            policy = Variable(actor(sb).squeeze(), requires_grad=False)
            prob = F.softmax(policy)
            argmax_acs[i] = th.argmax(prob).clone().detach()
            rand_acs[i] = Categorical(prob).sample().type(LongTensor)
        argmax_acs = argmax_acs.squeeze()
        rand_acs = rand_acs.squeeze()
        if eps == 0.0:
            return argmax_acs
        # TODO:此处的steps_done待定啥时候用
        self.steps_done += 1
        return th.stack([argmax_acs[i] if r > eps else rand_acs[i] for i, r in
                        enumerate(th.rand(self.n_agents))])


    def save(self, model_dir, train_step, save_cycle, time_now):
        """
        Save trained parameters of all agents into one file
        """
        num = str(train_step // save_cycle)
        model_dir = './model/' + time_now
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        for i in range(9):
            w = str(num)
            q = str(i)
            th.save(self.actors[i].state_dict(), model_dir + '/' + w + '_'+ q + 'actor_params.pkl')
            th.save(self.actors_target[i].state_dict(), model_dir + '/' + w + '_'+ q + 'actor_target_params.pkl')
            th.save(self.critics[i].state_dict(), model_dir + '/' + w + '_'+ q + 'critic_params.pkl')
            th.save(self.critics_target[i].state_dict(), model_dir + '/'+ w + '_' + q + 'critic_target_params.pkl')