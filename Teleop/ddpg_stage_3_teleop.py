#!/home/ubuntu/anaconda3/envs/tbtorch/bin/python3
# Authors: Junior Costa de Jesus #

import rospy
import os
import json
import numpy as np
import random
import time
import sys

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from collections import deque
from std_msgs.msg import Float32
from environment_stage_3 import Env
import torch
import torch.nn.functional as F
import gc
import torch.nn as nn
import math
from collections import deque
import copy
import time
# import mlflow
import pandas as pd
from environment_stage_3 import *


def get_teleop_velocity(cmd_vel):
    global action
    l_vel = cmd_vel.linear.x
    a_vel = cmd_vel.angular.z
    action = [l_vel, a_vel]
    #rospy.loginfo(velocity)
    

rospy.Subscriber('cmd_vel', Twist, queue_size=5, callback=get_teleop_velocity)

# (22.08.11 park) 사용 파라미터 한번에 관리하도록 구조 변경
# (22.07.21 park) reward 쉽게 변경하도록 모드 설정
reward_dict = {1: "Immediate, Original", 2: "Immediate, New", 3: "Accumulate, New"}
stage_dict = {1: "Vanilla", 2: "Fixed Obstacle", 3:"Moving Obstacle"}
reward_mode = 1
ensemble = 2
start_time = time.time()
is_training = True
exploration_decay_rate = 0.001
BATCH_SIZE = 2000  # 256 * 2
LEARNING_RATE = 0.001
GAMMA = 0.99
TAU = 0.001

MAX_EPISODES = 1001
MAX_STEPS = 300
MAX_BUFFER = 20000
rewards_all_episodes = []

EPS = 0.003 # Critic weight 초기화 관련 파라미터

ACTION_DIMENSION = 2
ACTION_V_MAX = 0.22  # m/s
ACTION_W_MAX = 2.    # rad/s

# (22.07.22 kwon) 성공/충돌 시 보상 변수로 설정
# goal reward / collision reward
goal_reward = 500
collision_reward = -550

# (22.08.11 park) Stage 별 파라미터 변경하도록 함
world = 'stage_3'
STATE_DIMENSION = 16

if reward_mode == 1:
    print("******************* reward mode : ", reward_dict[1], " ********************")
if reward_mode == 2:
    print("******************* reward mode : ", reward_dict[2], " ********************")
if reward_mode == 3:
    print("******************* reward mode : ", reward_dict[3], " ********************")


# ---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))


# ---Run agent---#

if is_training:
    var_v = ACTION_V_MAX * .5
    var_w = ACTION_W_MAX * 2 * .5
else:
    var_v = ACTION_V_MAX * 0.10
    var_w = ACTION_W_MAX * 0.10

print('State Dimensions: ' + str(STATE_DIMENSION))
print('Action Dimensions: ' + str(ACTION_DIMENSION))
print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s')

reward_list = []
new_reward_list = []

if __name__ == '__main__':
    rospy.init_node('ddpg_stage_3')
    before_training = 1
    env = Env(action_dim=ACTION_DIMENSION)
    goal = False
   # for ensemble in range(10):

    offline_data = {}
    offline_s = []
    offline_a = []
    offline_r = []
    offline_d = []
    offline_n_s = []
    offline_count = 0
    offline_data_final = False
    
    past_action = np.zeros(ACTION_DIMENSION)
    
    for ep in range(MAX_EPISODES):
        if ep == MAX_EPISODES - 1:
            offline_data_final = True
        done = False
        if not goal:
            state, prev_cord = env.reset()
        goal = False
        
        print("episode ", ep, " start")

        original_rewards = 0.
        new_rewards = 0.
        temp_list = []

        for step in range(MAX_STEPS):
            state = np.float32(state)
            
            # (22.07.21 park) Reward 명 변경
            next_state, original_reward, new_reward, done, goalbox, cord = env.step(action, past_action)
            
            offline_s.append(state)
            offline_a.append(action)
            offline_r.append(original_reward)
            offline_d.append(done)
            offline_n_s.append(next_state)
            
            print("State: ", state, " Next state: ", next_state, " original_reward: ", original_rewards, " done: ", done)
            # print('action', action,'r',reward)
            past_action = action

            next_state = np.float32(next_state)

             # (22.07.22. kwon) reward_mode 별 ram 에 추가해야하는 immediate reward 로 list 생성 -
            reward_arg = [original_reward, new_reward, new_rewards]


            state = copy.deepcopy(next_state)
            prev_cord = copy.deepcopy(cord)
            
            if len(offline_s) > 1000000:
                offline_data = {'state' : offline_s, 'action' : offline_a, 'reward' : offline_r, 'done' : offline_d, 'next_state' : offline_n_s}
                df_name = "offline_datasets ep" + str(ep) + " Ensemble " + str(ensemble) + " data num " + str(offline_count) + ".csv" 
                df =  pd.DataFrame(offline_data)
                if not os.path.exists(dirPath + "/Offline Datasets/" + world):
                    os.makedirs(dirPath + "/Offline Datasets/" + world)
                df5.to_csv(dirPath + "/Offline Datasets/" + world + "/" +df_name) 
         
                offline_count += 1

                offline_s = []
                offline_a = []
                offline_r = []
                offline_d = []
                offline_n_s = []
            
            if done or step == MAX_STEPS - 1 or goal:
                print('Original_reward per ep: ' + str(original_rewards))
                print('*\nbreak step: ' + str(step) + '\n*')

                break

        print("saving completed!")
        if offline_data_final:
            offline_data = {'state' : offline_s, 'action' : offline_a, 'reward' : offline_r, 'done' : offline_d, 'next_state' : offline_n_s}
            df_name = "offline_datasets ep" + str(ep) + " Ensemble " + str(ensemble) + " data num " + str(offline_count) + ".csv" 
            df =  pd.DataFrame(offline_data)
            if not os.path.exists(dirPath + "/Offline Datasets/" + world):
                os.makedirs(dirPath + "/Offline Datasets/" + world)
            df5.to_csv(dirPath + "/Offline Datasets/" + world + "/" +df_name) 
            
        #if ep % 1000 == 0 and ep > 0:
            #df1_name = "reward_ep" + str(ep) + " Ensemble_" + str(ensemble) + ".csv"
            #df2_name = "new_rewad_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"
            #df3_name = "actor_loss_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"
            #df4_name = "critic_loss_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"

            print("data saving...")
            #df1 = pd.DataFrame(reward_list, columns=['reward'])
            #df2 = pd.DataFrame(new_reward_list, columns=['reward'])
            #df3 = pd.DataFrame(actor_loss, columns=['loss'])
            #df4 = pd.DataFrame(critic_loss, columns=['loss'])
                            
            # (22.07.28. park) 폴더 경로 자동으로 설정
            # (22.08.11. park) World별 Reward 따로 저장하도록 변경
            #if not os.path.exists(dirPath + "/Rewards/" + world + '/Original Rewards/'):
            #    os.makedirs(dirPath + "/Rewards/" + world + '/Original Rewards/')
            #if not os.path.exists(dirPath + "/Rewards/" + world + '/New Rewards/'):
            #    os.makedirs(dirPath + "/Rewards/" + world + '/New Rewards/')
            #if not os.path.exists(dirPath + "/Loss/" + world + '/Critic Loss/'):
            #    os.makedirs(dirPath + "/Loss/" + world + '/Critic Loss/')
            #if not os.path.exists(dirPath + "/Loss/" + world + '/Actor Loss/'):
            #    os.makedirs(dirPath + "/Loss/" + world + '/Actor Loss/')
                
            #df1.to_csv(dirPath + "/Rewards/" + world + '/Original Rewards/' + df1_name)
            #df2.to_csv(dirPath + "/Rewards/" + world + '/New Rewards/' + df2_name)
            #df3.to_csv(dirPath + "/Loss/" + world + '/Actor Loss/' + df3_name)
            #df4.to_csv(dirPath + "/Loss/" + world + '/Critic Loss/' + df4_name)

            print("data saving completed!!")

print('Completed Training')
print('processing time : ', time.time() - start_time)
