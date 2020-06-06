'''
Interactions with Sumo, get/set values from Sumo, pass it to RL agents
log the Simulation Data
'''

import logging
import numpy as np
import subprocess
import time
import xml.etree.cElementTree as ET
from sys import platform
import os
import sys
import copy

DEFAULT_PORT = 8000
SEC_IN_MS = 1000

REALNET_REWARD_NORM = 20

###### Please Specify the location of your traci module

if platform == "linux" or platform == "linux2":# this is linux
    os.environ['SUMO_HOME'] = '/usr/share/sumo'
    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
	            os.path.join(os.environ["SUMO_HOME"], "tools")
	        )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

elif platform == "win32":
    #os.environ['SUMO_HOME'] = 'C:\\Program Files (x86)\\DLR\\Sumo'
    os.environ['SUMO_HOME'] = 'D:\\SUMO'

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
elif platform =='darwin':
    os.environ['SUMO_HOME'] = "/Users/{0}/sumo/sumo-git".format(os.getlogin())

    try:
        import traci
        import traci.constants as tc
    except ImportError:
        if "SUMO_HOME" in os.environ:
            print(os.path.join(os.environ["SUMO_HOME"], "tools"))
            sys.path.append(
                os.path.join(os.environ["SUMO_HOME"], "tools")
            )
            try:
                import traci
                import traci.constants as tc
            except ImportError:
                raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")
        else:
            raise EnvironmentError("Please set SUMO_HOME environment variable or install traci as python module!")

else:
    sys.exit("platform error")


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

# 交通相位列表
NSG = "GGGrrrrrGGGrrrrr".replace(" ", "")
NSR = "yyyryrrryyyryrrr".replace(" ", "")
NSLG = "rrrGrrrrrrrGrrrr".replace(" ", "")
NSLR = "yrryyrrryrryyrrr".replace(" ", "")
WEG = "rrrrGGGrrrrrGGGr".replace(" ", "")
WER = "yrrryyyryrrryyyr".replace(" ", "")
WELG = "rrrrrrrGrrrrrrrG".replace(" ", "")
WELR = "yrrryrryyrrryrry".replace(" ", "")
controlSignal = (NSG, NSLG, WEG, WELG)

# min_phase_time
min_phase_time = 10

#开始sumo
def start_sumo(sumo_cmd_str):
    traci.start(sumo_cmd_str)

#中止sumo
def end_sumo():
    traci.close()

def get_current_phase(Tid):
    return traci.getPhase(Tid)

#获取当前时间
def get_current_time():
    return traci.simulation.getTime()

#根据Lanes获取queue length
def get_queue_length(listLanes):
    queue_length = 0
    for lane in listLanes:
        #返回给定车道上last time step停止的车辆总数，停止：<0.1m/s
        queue_length += traci.lane.getLastStepHaltingNumber(lane)
    return queue_length

#根据Lanes获取queue length
def get_waiting_time(listLanes):
    waiting_time = 0
    for lane in listLanes:
        waiting_time += traci.lane.getWaitingTime(str(lane)) / 60.0
    return waiting_time

'''
获取state，里面的子项为："node_name":{"queue_length":queue_length, "waiting_time":waiting_time}
获取全部的queue length 和waiting time
输入node_names
'''
def get_state(node_names):
    overall_queue_length = 0
    overall_waitimg_time = 0
    state = {}
    for node_name in node_names:
        list_lane = lanes_nodes.get(node_name)
        queue_length = get_queue_length(list_lane)
        waiting_time = get_waiting_time(list_lane)
        state[node_name] = {}
        state[node_name]["queue_length"] = queue_length
        state[node_name]["waiting_time"] = waiting_time
        overall_queue_length += queue_length
        overall_waitimg_time += waiting_time
    return state, overall_queue_length, overall_waitimg_time

# 改变当前相位的函数
# 设置红绿灯相位
# 设置0号路口红绿灯的相位
# input:当前相位
# output:下一相位，下一相位时间重叠
def changeTrafficLight_7(current_phase, node_name):  # [WNG_ESG_WSG_ENG_NWG_SEG]
    # phases=["WNG_ESG_WSG_ENG_NWG_SEG","EWG_WEG_WSG_ENG_NWG_SEG","NSG_NEG_SNG_SWG_WSG_ENG_NWG_SEG"]
    next_phase = (current_phase + 1) % len(controlSignal)
    next_phase_time_eclipsed = 0
    traci.trafficlight.setPhase(node_name, next_phase)
    return next_phase, next_phase_time_eclipsed

# 记录
# def log_reward(action, reward_info_dict, file_name, timestamp, rewards_detail_dict_list, node_name):
#     reward, reward_detail_dict = get_reward_from_sumo(action, reward_info_dict, node_name)
#     list_reward_keys = np.sort(list(reward_detail_dict.keys()))
#     reward_str = "{0}, {1}".format(timestamp,action)
#     for reward_key in list_reward_keys:
#         reward_str = reward_str + ", {0}".format(reward_detail_dict[reward_key][2])
#     reward_str += '\n'

#     fp = open(file_name, "a")
#     fp.write(reward_str)
#     fp.close()
#     rewards_detail_dict_list.append(reward_detail_dict)

# def get_reward_from_sumo(action, rewards_info_dict, node_name):
#     reward_detail_dict = copy.deepcopy(rewards_info_dict)
#     list_lane = lists_lane.get(node_name)
#     queue_length = get_queue_length(list_lane)
#     waiting_time = get_waiting_time(list_lane)
#     reward_detail_dict["queue_length"].append(queue_length)
#     reward_detail_dict["waiting_time"].append(waiting_time)
    
#     for k, v in reward_detail_dict.items():
#         if v[0]:
#             reward += v[1] * v[2]
#     return reward, reward_detail_dict

# 记录全部路口的reward信息
# def log_all_reward( reward_info_dict, file_name, timestamp, rewards_detail_dict_list, node_names):
#     reward, reward_detail_dict = get_all_reward_from_sumo(reward_info_dict, node_names)
#     list_reward_keys = np.sort(list(reward_detail_dict.keys()))
#     reward_str = "{0}".format(timestamp)
#     for reward_key in list_reward_keys:
#         reward_str = reward_str + ", {0}".format(reward_detail_dict[reward_key][2])
#     reward_str += '\n'

#     fp = open(file_name, "a")
#     fp.write(reward_str)
#     fp.close()
#     rewards_detail_dict_list.append(reward_detail_dict)

# def get_all_reward_from_sumo(reward_info_dict, node_names):
#     reward_detail_dicts = copy.deepcopy(reward_info_dict)
#     _, overall_queue_length, overall_waitimg_time = get_state(node_names)
#     reward_detail_dicts["queue_length"].append(overall_queue_length)
#     reward_detail_dicts["waiting_time"].append(overall_waitimg_time)

#     for k, v in reward_detail_dicts.items():
#         if v[0]:
#             reward += v[1] * v[2]
#     return reward, reward_detail_dicts

