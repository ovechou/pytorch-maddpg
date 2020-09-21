# -*- coding: utf-8 -*-

'''
@author: OveChou

Interacting with basic_controller.py and env.py

1) retriving values from env.py

2) update state

3) controling logic

'''

import envs.env as env
import json
import os
import traci

import torch

import time

min_time = 30


ql_weight = -0.5
wt_weight = -0.25

# init node
node_names = ["nl1", "nl2", "nl3",
            "nm1", "nm2", "nm3",
            "nr1", "nr2", "nr3"]

# init lane
list_lane_nl1 = ["nl11_nl1_0", "nl11_nl1_1", "nl11_nl1_2", "nm1_nl1_0", "nm1_nl1_1", "nm1_nl1_2", "nl2_nl1_0", "nl2_nl1_1", "nl2_nl1_2", "nl10_nl1_0", "nl10_nl1_1", "nl10_nl1_2"]
list_lane_nl2 = ["nl1_nl2_0", "nl1_nl2_1", "nl1_nl2_2", "nm2_nl2_0", "nm2_nl2_1", "nm2_nl2_2", "nl20_nl2_0", "nl20_nl2_1", "nl20_nl2_2", "nl3_nl2_0", "nl3_nl2_1",  "nl3_nl2_2"]
list_lane_nl3 = ["nl2_nl3_0", "nl2_nl3_1", "nl2_nl3_2", "nm3_nl3_0", "nm3_nl3_1", "nm3_nl3_2", "nl31_nl3_0", "nl31_nl3_1", "nl31_nl3_2", "nl30_nl3_0", "nl30_nl3_1", "nl30_nl3_2"]
list_lane_nm1 = ["nm11_nm1_0", "nm11_nm1_1", "nm11_nm1_2", "nr1_nm1_0", "nr1_nm1_1", "nr1_nm1_2", "nm2_nm1_0", "nm2_nm1_1", "nm2_nm1_2", "nl1_nm1_0", "nl1_nm1_1", "nl1_nm1_2"]
list_lane_nm2 = ["nm1_nm2_0", "nm1_nm2_1", "nm1_nm2_2", "nr2_nm2_0", "nr2_nm2_1", "nr2_nm2_2", "nm3_nm2_0", "nm3_nm2_1", "nm3_nm2_2", "nl2_nm2_0", "nl2_nm2_1", "nl2_nm2_2"]
list_lane_nm3 = ["nm2_nm3_0", "nm2_nm3_1", "nm2_nm3_2", "nr3_nm3_0", "nr3_nm3_1", "nr3_nm3_2", "nm31_nm3_0", "nm31_nm3_1", "nm31_nm3_2", "nl3_nm3_0", "nl3_nm3_1", "nl3_nm3_2"]
list_lane_nr1 = ["nr11_nr1_0", "nr11_nr1_1", "nr11_nr1_2", "nr10_nr1_0", "nr10_nr1_1", "nr10_nr1_2", "nr2_nr1_0", "nr2_nr1_1", "nr2_nr1_2", "nm1_nr1_0", "nm1_nr1_1", "nm1_nr1_2"]
list_lane_nr2 = ["nr1_nr2_0", "nr1_nr2_1", "nr1_nr2_2", "nr20_nr2_0", "nr20_nr2_1", "nr20_nr2_2", "nr3_nr2_0", "nr3_nr2_1", "nr3_nr2_2", "nm2_nr2_0", "nm2_nr2_1", "nm2_nr2_2"]
list_lane_nr3 = ["nr2_nr3_0", "nr2_nr3_1", "nr2_nr3_2", "nr30_nr3_0", "nr30_nr3_1", "nr30_nr3_2", "nr31_nr3_0", "nr31_nr3_1", "nr31_nr3_2", "nm3_nr3_0", "nm3_nr3_1", "nm3_nr3_2"]

# "nl3_nl2_0", "nl20_nl2_0"

# lists of lane
lanes_nodes = {"nl1":list_lane_nl1, "nl2":list_lane_nl2, "nl3":list_lane_nl3,
              "nm1":list_lane_nm1, "nm2":list_lane_nm2, "nm3":list_lane_nm3,
              "nr1":list_lane_nr1, "nr2":list_lane_nr2, "nr3":list_lane_nr3}

counts = {"nl1":0, "nl2":0, "nl3":0,
          "nm1":0, "nm2":0, "nm3":0,
          "nr1":0, "nr2":0, "nr3":0}

red_times = {"nl1":0, "nl2":0, "nl3":0,
          "nm1":0, "nm2":0, "nm3":0,
          "nr1":0, "nr2":0, "nr3":0}

real_actions = {"nl1":0, "nl2":0, "nl3":0,
          "nm1":0, "nm2":0, "nm3":0,
          "nr1":0, "nr2":0, "nr3":0}

record_lane_nl1 = ["nm1_nl1_0", "nm1_nl1_1", "nm1_nl1_2", "nl2_nl1_0", "nl2_nl1_1", "nl2_nl1_2"]
record_lane_nl2 = ["nl1_nl2_0", "nl1_nl2_1", "nl1_nl2_2", "nm2_nl2_0", "nm2_nl2_1", "nm2_nl2_2", "nl3_nl2_0", "nl3_nl2_1",  "nl3_nl2_2"]
record_lane_nl3 = ["nl2_nl3_0", "nl2_nl3_1", "nl2_nl3_2", "nm3_nl3_0", "nm3_nl3_1", "nm3_nl3_2"]
record_lane_nm1 = ["nr1_nm1_0", "nr1_nm1_1", "nr1_nm1_2", "nm2_nm1_0", "nm2_nm1_1", "nm2_nm1_2", "nl1_nm1_0", "nl1_nm1_1", "nl1_nm1_2"]
record_lane_nm2 = ["nm1_nm2_0", "nm1_nm2_1", "nm1_nm2_2", "nr2_nm2_0", "nr2_nm2_1", "nr2_nm2_2", "nm3_nm2_0", "nm3_nm2_1", "nm3_nm2_2", "nl2_nm2_0", "nl2_nm2_1", "nl2_nm2_2"]
record_lane_nm3 = ["nm2_nm3_0", "nm2_nm3_1", "nm2_nm3_2", "nr3_nm3_0", "nr3_nm3_1", "nr3_nm3_2", "nl3_nm3_0", "nl3_nm3_1", "nl3_nm3_2"]
record_lane_nr1 = ["nr2_nr1_0", "nr2_nr1_1", "nr2_nr1_2", "nm1_nr1_0", "nm1_nr1_1", "nm1_nr1_2"]
record_lane_nr2 = ["nr1_nr2_0", "nr1_nr2_1", "nr1_nr2_2", "nr3_nr2_0", "nr3_nr2_1", "nr3_nr2_2", "nm2_nr2_0", "nm2_nr2_1", "nm2_nr2_2"]
record_lane_nr3 = ["nr2_nr3_0", "nr2_nr3_1", "nr2_nr3_2", "nm3_nr3_0", "nm3_nr3_1", "nm3_nr3_2"]

record_nodes = {"nl1":record_lane_nl1, "nl2":record_lane_nl2, "nl3":record_lane_nl3,
              "nm1":record_lane_nm1, "nm2":record_lane_nm2, "nm3":record_lane_nm3,
              "nr1":record_lane_nr1, "nr2":record_lane_nr2, "nr3":record_lane_nr3}

class SumoAgent:
    def __init__(self, sumo_cmd_str, num_agents):
        # self.path_set = path_set

        self.current_phase = 0
        
        env.start_sumo(sumo_cmd_str)

    def start_sumo(self, sumo_cmd_str):
        env.start_sumo(sumo_cmd_str)

    def end_sumo(self):
        env.end_sumo()

    def load_conf(self, conf_file):
        dic_paras = json.read(open(conf_file, "r"))
        return dic_paras

    def get_current_time(self):
        return env.get_current_time()


    def get_obs(self):
        obs =[]
        for node_name in node_names:
            local_obs = []
            listlanes = lanes_nodes[node_name]
            queue_length = self.get_queue_length(listlanes)
            waiting_time = self.get_waiting_time(listlanes)
            local_obs.append(queue_length)
            local_obs.append(waiting_time)
            obs.append(local_obs)
        return obs

    def get_state(self, node_names):
        state = []
        _, total_queue_length, total_waiting_time = env.get_state(node_names)
        state.append(total_queue_length)
        state.append(total_waiting_time)
        return state

    #根据Lanes获取queue length
    def get_queue_length(self, listLanes):
        queue_length = 0
        for lane in listLanes:
            #返回给定车道上last time step停止的车辆总数，停止：<0.1m/s
            queue_length += traci.lane.getLastStepHaltingNumber(lane)
        return queue_length

    #根据Lanes获取queue length
    def get_waiting_time(self, listLanes):
        waiting_time = 0
        for lane in listLanes:
            waiting_time += traci.lane.getWaitingTime(str(lane)) / 60.0
        return waiting_time


    def step(self, actions, time_now):
        i = 0
        rewards = []
        timestamp = env.get_current_time()
        for node_name in node_names:
            local_reward = []
            reward = 0
            count = counts[node_name]
            action = actions[i]
            list_lanes = lanes_nodes[node_name]
            queue_length = self.get_queue_length(list_lanes)
            waiting_time = self.get_waiting_time(list_lanes)
            current_phase = traci.trafficlight.getPhase(node_name)
            count += 1
            if count < 15:
                if action == 1:
                    current_phase, _ = env.changeTrafficLight_7(current_phase, node_name)
                    reward = ql_weight * queue_length + wt_weight * waiting_time - 100
                    count = 0
                else:
                    reward = ql_weight * queue_length + wt_weight * waiting_time
            else:
                reward = ql_weight * queue_length + wt_weight * waiting_time
                if action == 1:
                    current_phase, _ = env.changeTrafficLight_7(current_phase, node_name)
                    count = 0
            counts[node_name] = count
            file_name = "ove/record/Maddpg/{0}".format(time_now)
            if not os.path.exists(file_name):
                os.makedirs(file_name)
            file_name = file_name + '/'
            reward_str = "{0}, {1}".format(timestamp, action)
            reward_str = reward_str + ", {0}".format(queue_length)
            reward_str = reward_str + ", {0}".format(waiting_time)
            reward_str = reward_str + ", {0}".format(current_phase)
            reward_str += '\n'
            f_log_rewards = os.path.join(file_name,'log_rewards_{0}.txt'.format(node_name))
            if not os.path.exists(f_log_rewards):
                f = open(f_log_rewards, 'w')
                head_str = "count,action,queue_length,waiting_time,current_phase,local_queue_length"
                head_str = head_str + '\n'
                f.write(head_str)
                f.close()
            fp = open(f_log_rewards, "a")
            fp.write(reward_str)
            fp.close
            local_reward.append(reward)
            rewards.append(local_reward)
            i += 1
        traci.simulationStep()
        return rewards