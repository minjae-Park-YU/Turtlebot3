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
import torch.optim as optim
from torch.distributions import Normal
import gc
import torch.nn as nn
import math
from collections import deque
import copy
import pandas as pd

ensemble = 2
H_weight = 1
R_weight = 1
goal_reward = 500
collision_reward = -550
#---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))

#****************************************************


class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
    
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim, init_w=3e-3):
        super(ValueNetwork, self).__init__()
        
        self.linear1 = nn.Linear(state_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2_3 = nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = nn.Linear(hidden_dim, 1)
        
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear2_3.weight.data.uniform_(-init_w, init_w)
        self.linear2_3.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = mish(self.linear1(state))
        x = mish(self.linear2(x))
        x = mish(self.linear2_3(x))
        x = self.linear3(x)
        return x
        
        
class SoftQNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(SoftQNetwork, self).__init__()
        
        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2_3 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        self.linear2_3.weight.data.uniform_(-init_w, init_w)
        self.linear2_3.bias.data.uniform_(-init_w, init_w)
        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = mish(self.linear1(x))
        x = mish(self.linear2(x))
        x = mish(self.linear2_3(x))
        x = self.linear3(x)
        return x
        
        
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3, log_std_min=-10, log_std_max=2):
        super(PolicyNetwork, self).__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear1.weight.data.uniform_(-init_w, init_w)
        self.linear1.bias.data.uniform_(-init_w, init_w)
        
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear2.weight.data.uniform_(-init_w, init_w)
        self.linear2.bias.data.uniform_(-init_w, init_w)
        
        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.mean_linear.weight.data.uniform_(-init_w, init_w)
        self.mean_linear.bias.data.uniform_(-init_w, init_w)
        
        self.log_std_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.bias.data.uniform_(-init_w, init_w)
        
    def forward(self, state):
        x = mish(self.linear1(state))
        x = mish(self.linear2(x))
        
        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def evaluate(self, state, epsilon=1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z = normal.sample()
        action = torch.tanh(z)
        
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(-1, keepdim=True)
        
        return action, log_prob, z, mean, log_std
        
    
    def get_action(self, state, exploitation=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = Normal(mean, std)
        z      = normal.sample()
        action = torch.tanh(z)
        if exploitation:
            action = torch.tanh(mean)
        #action = z.detach().numpy()
        
        action  = action.detach().numpy()
        return action[0]


def soft_q_update(batch_size, 
           gamma=0.99,
           mean_lambda=1e-3,
           std_lambda=1e-3,
           z_lambda=0.0,
           soft_tau=1e-2,
          ):
    state, action, reward, next_state, done = replay_buffer.sample(batch_size)

    state      = torch.FloatTensor(state)
    next_state = torch.FloatTensor(next_state)
    action     = torch.FloatTensor(action)
    reward     = torch.FloatTensor(reward).unsqueeze(1)
    done       = torch.FloatTensor(np.float32(done)).unsqueeze(1)
    #print('done', done)

    expected_q_value = soft_q_net(state, action)
    expected_value   = value_net(state)
    new_action, log_prob, z, mean, log_std = policy_net.evaluate(state)


    target_value = target_value_net(next_state)
    next_q_value = reward + (1 - done) * gamma * target_value
    q_value_loss = soft_q_criterion(expected_q_value, next_q_value.detach())

    expected_new_q_value = soft_q_net(state, new_action)
    next_value = expected_new_q_value - log_prob
    value_loss = value_criterion(expected_value, next_value.detach())

    log_prob_target = expected_new_q_value - expected_value
    policy_loss = (log_prob * (log_prob - log_prob_target).detach()).mean()
    

    mean_loss = mean_lambda * mean.pow(2).mean()
    std_loss  = std_lambda  * log_std.pow(2).mean()
    z_loss    = z_lambda    * z.pow(2).sum(1).mean()

    policy_loss += mean_loss + std_loss + z_loss

    soft_q_optimizer.zero_grad()
    q_value_loss.backward()
    soft_q_optimizer.step()

    value_optimizer.zero_grad()
    value_loss.backward()
    value_optimizer.step()

    policy_optimizer.zero_grad()
    policy_loss.backward()
    policy_optimizer.step()
    
    return q_value_loss.detach().numpy().tolist(), value_loss.detach().numpy().tolist(), policy_loss.detach().numpy().tolist()
    
    
    for target_param, param in zip(target_value_net.parameters(), value_net.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - soft_tau) + param.data * soft_tau
        )


#---Mish Activation Function---#
def mish(x):
    '''
        Mish: A Self Regularized Non-Monotonic Neural Activation Function
        https://arxiv.org/abs/1908.08681v1
        implemented for PyTorch / FastAI by lessw2020
        https://github.com/lessw2020/mish
        param:
            x: output of a layer of a neural network
        return: mish activation function
    '''
    return torch.clamp(x*(torch.tanh(F.softplus(x))),max=6)

#----------------------------------------------------------

action_dim = 2
state_dim  = 16
hidden_dim = 500
ACTION_V_MIN = 0.0 # m/s
ACTION_W_MIN = -2. # rad/s
ACTION_V_MAX = 0.22 # m/s
ACTION_W_MAX = 2. # rad/s
world = 'stage_3'

value_net        = ValueNetwork(state_dim, hidden_dim)
target_value_net = ValueNetwork(state_dim, hidden_dim)

soft_q_net = SoftQNetwork(state_dim, action_dim, hidden_dim)
policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim)

def hard_update(target,source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

value_criterion  = nn.MSELoss()
soft_q_criterion = nn.MSELoss()

value_lr  = 3e-4
soft_q_lr = 3e-4
policy_lr = 3e-4

value_optimizer  = optim.Adam(value_net.parameters(), lr=value_lr)
soft_q_optimizer = optim.Adam(soft_q_net.parameters(), lr=soft_q_lr)
policy_optimizer = optim.Adam(policy_net.parameters(), lr=policy_lr)


replay_buffer_size = 50000
replay_buffer = ReplayBuffer(replay_buffer_size)

print('State Dimensions: ' + str(state_dim))
print('Action Dimensions: ' + str(action_dim))
print('Action Max: ' + str(ACTION_V_MAX) + ' m/s and ' + str(ACTION_W_MAX) + ' rad/s')
#-----------------------------------------------------
def save_models(episode_count):
    if not os.path.exists(dirPath + '/SAC_Models/' + world + '/' + str(ensemble) + '/'):
                os.makedirs(dirPath + '/SAC_Models/' + world + '/' + str(ensemble) + '/')
    torch.save(policy_net.state_dict(), dirPath + '/SAC_Models/' + world + '/'+  str(ensemble) + '/' + str(episode_count)+ '_policy_net.pth')
    torch.save(value_net.state_dict(), dirPath + '/SAC_Models/' + world + '/' +  str(ensemble) + '/'+ str(episode_count)+ 'value_net.pth')
    torch.save(soft_q_net.state_dict(), dirPath + '/SAC_Models/' + world + '/' +  str(ensemble) + '/'+ str(episode_count)+ 'soft_q_net.pth')
    torch.save(target_value_net.state_dict(), dirPath + '/SAC_Models/' + world + '/'+  str(ensemble) + '/' + str(episode_count)+ 'target_value_net.pth')
    print("====================================")
    print("Model has been saved...")
    print("====================================")

def load_models(episode):
    policy_net.load_state_dict(torch.load(dirPath + '/SAC_Models/' + world + '/'+  str(ensemble) + '/' + str(episode_count)+ '_policy_net.pth'))
    value_net.load_state_dict(torch.load(dirPath + '/SAC_Models/' + world + '/' +  str(ensemble) + '/'+ str(episode_count)+ 'value_net.pth'))
    soft_q_net.load_state_dict(torch.load(dirPath + '/SAC_Models/' + world + '/' +  str(ensemble) + '/'+ str(episode_count)+ 'soft_q_net.pth'))
    target_value_net.load_state_dict(torch.load(dirPath + '/SAC_Models/' + world + '/'+  str(ensemble) + '/' + str(episode_count)+ 'target_value_net.pth'))
    print('***Models load***')

def newReward(prev_cord, cord, step, reward_mode, terminal_state, state, past_min_LDS):  # for HER
    prev_x = prev_cord[0]
    prev_y = prev_cord[1]
    current_x = cord[0]
    current_y = cord[1]
    goal_x = cord[2]
    goal_y = cord[3]
    newgoal_x = cord[4]
    newgoal_y = cord[5]
    # print("new_goal in newReward : " + str(newgoal_x) + ", " + str(newgoal_y))
    newgoal_box = False
    # print("current x : ", current_x)
    # print("current y : ", current_y)
    # print("goal x : ", newgoal_x)
    # print("goal y : ", newgoal_y)

    past_distance = round(math.hypot(newgoal_x - prev_x, newgoal_y - prev_y), 2)
    current_distance = round(math.hypot(newgoal_x - current_x, newgoal_y - current_y), 2)
    # print("newReward_current_distance : " + str(current_distance))
    distance_rate = (past_distance - current_distance)

    current_distance = state[-3]
    LDS_values = state[0:10]
    sorted_LDS_values = sorted(LDS_values)
    min_LDS = sorted_LDS_values[0]
        
    if past_min_LDS > 0:
        if past_min_LDS - min_LDS > 0:
            obstacle_penalty = -550 * np.exp(-70*(min_LDS-0.135))
        else:
            obstacle_penalty = 550 * np.exp(-70*(min_LDS-0.135))
    else:
        obstacle_penalty = 0
        
    # (22.07.21. park) Reward mode에 따라 HER에 적용되는 Reward 계산방식 작성
    if reward_mode == 1:
        if distance_rate > 0:
            reward = 200. * distance_rate
        if distance_rate == 0:
            reward = 0.
        if distance_rate <= 0:
            reward = -8.

    elif reward_mode == 2:
        reward = -1  # 1 Step 마다 -1의 Reward


    elif reward_mode == 3:
        reward = -1 * step  # 1 Step 마다 -1의 Reward
        
    reward += obstacle_penalty

	# (22.07.22. kwon) reward mode 에 상관없이 아래 조건문이 사용되므로 조건문 밖에서 한 번만 실행하는 것으로 변경
    #if current_distance < 0.15:  # 목적지에 도달했을 시 500점 추가, HER을 적용하기 위해
		# (22.07.20. kwon) 실제 보상과 일치시키기 위해 reward += 500 으로 수정
		# (22.07.22. kwon) 위 사항 재수정. immediate reward 이므로 += 에서 = 으로 수정
    if terminal_state:
        reward = goal_reward  # 충돌 직전의 좌표를 목적지로 생각하기 때문에, 충돌 리워드는 없음
        newgoal_box = True

    goal_angle = math.atan2(goal_y - current_y, goal_x - current_x)
    new_goal_angle = math.atan2(newgoal_y - current_y, newgoal_x - current_x)

    return reward, current_distance, new_goal_angle - goal_angle, newgoal_box


# (22.07.20 kwon) setHER 함수 주석 추가
# temp_list : HER 설정을 위한 list
# step : 과거 몇 step 까지 HER 에 사용할지를 결정하는 변수
def setHER(temp_list, HER_distance, reward_mode, ram):
    # (22.07.20 kwon) 조건문 주석 추가
    # temp_list[-1][3] : 충돌인 경우 확인 (충돌 = True)
    # len(temp_list[:]) > step   :  (전체스텝 - step) 에 대해 HER 적용이 가능한지 확인
    # if temp_list[-1][3] and len(temp_list[:]) > step:
    print("modified goal data saving...")
    # (22.07.20. kwon) new_goal 주석 추가
    terminal_state = False
    current_distance = 0
    past_min_LDS = 0
    for i in range(len(temp_list)-1, -1, -1):
        #print(abs(temp_list[i][6][0][0]))
        #print(abs(temp_list[i-1][6][0][0]))
        current_distance += math.hypot(abs(abs(temp_list[i][6][0][0]) - abs(temp_list[i-1][6][0][0])), abs(abs(temp_list[i][6][0][1]) - abs(temp_list[i-1][6][0][1])))
        #print("Current distance: ", current_distance)

        if current_distance > HER_distance: 
            new_goal_x = temp_list[i-1][6][0][0]  # 전체스텝 - step 의 위치를 new_goal 로 설정
            new_goal_y = temp_list[i-1][6][0][1]
            terminal_count = i
            break
        
    #for i in range(len(temp_list[:]) - step):  # 전체스텝 - step 만큼 반복
    if current_distance > HER_distance:
        print("new_goal : " + str(new_goal_x) + ", " + str(new_goal_y))
        for i in range(terminal_count):
            if i == terminal_count-1:
                terminal_state = True
            st = temp_list[i][0]  # state
            act = temp_list[i][1]  # action
            n_st = temp_list[i][2]  # next state
            dn = temp_list[i][3]  # done
            prev_cord = temp_list[i][4]  # prev_cord
            gb = temp_list[i][5]  # goalbox
            # (22.07.22 kwon) cd 및 n_cd 생성방법 변경. 기존 방식의 경우 각 리스트가 초기화 되지 않는 문제 발생
            cd = list(range(6))
            n_cd = list(range(6))

            cd[0] = temp_list[i][6][0][0]
            cd[1] = temp_list[i][6][0][1]
            cd[2] = temp_list[i][6][0][2]
            cd[3] = temp_list[i][6][0][3]
            cd[4] = new_goal_x
            cd[5] = new_goal_y

            # print("new_goal in cd : ", cd)

            # print("Cd : ", temp_list[i + 1][6][0])
            # print("n_cd : ", n_cd[0])

            n_cd[0] = temp_list[i + 1][6][0][0]
            n_cd[1] = temp_list[i + 1][6][0][1]
            n_cd[2] = temp_list[i + 1][6][0][2]
            n_cd[3] = temp_list[i + 1][6][0][3]
            n_cd[4] = new_goal_x
            n_cd[5] = new_goal_y

            new_reward, current_distance, diff_goal_angle, n_goal_box = newReward(prev_cord, cd, i, reward_mode, terminal_state, st, past_min_LDS)
            st[-1] = current_distance
            st[-2] = diff_goal_angle + st[-2]
                
            _, n_current_distance, n_diff_goal_angle, _ = newReward(temp_list[i + 1][4], n_cd, i + 1, reward_mode, terminal_state, st, past_min_LDS)
            n_st[-1] = n_current_distance
            n_st[-2] = n_diff_goal_angle + n_st[-2]
            
            past_min_LDS = min(st[0:SCAN_RANGE])
            if n_goal_box:
                print('setHER goal step : ' + str(i))  # (22.07.22 kwon) HER 제대로 되었는지 확인을 위함
                for k in range(3):
                    ram.push(st, act, new_reward * H_weight, n_st, dn)
                break
            else:
                ram.push(st, act, new_reward * H_weight, n_st, dn)
            del cd, n_cd


#****************************
is_training = True

SCAN_RANGE = 10
#load_models(120)   
hard_update(target_value_net, value_net)
max_episodes  = 1001
MAX_STEPS   = 300
rewards     = []
batch_size  = 256
reward_mode = 1



#----------------------------------------
def action_unnormalized(action, high, low):
    action = low + (action + 1.0) * 0.5 * (high - low)
    action = np.clip(action, low, high)
    return action
#**********************************


if __name__ == '__main__':
    rospy.init_node('sac_stage_3')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    env = Env()
    before_training = 4
    past_action = np.array([0.,0.])
    goal = False
    ep_new_rewards = []
    ep_original_rewards = []
    training_count = 0
    ep_soft_q_loss = []
    ep_value_loss = []
    ep_policy_loss = []
    
    for ep in range(max_episodes):
        new_rewards = 0
        original_rewards = 0
        done = False
        if not goal:
            state, prev_cord = env.reset()
        goal = False
        
        if is_training and ep%2 == 0 and len(replay_buffer) > before_training*batch_size:
            print('Episode: ' + str(ep) + ' training')
        else:
            if len(replay_buffer) > before_training*batch_size:
                print('Episode: ' + str(ep) + ' evaluating')
            else:
                print('Episode: ' + str(ep) + ' adding to memory')

        rewards_current_episode = 0.
        
        temp_list = []
        
        for step in range(MAX_STEPS):
            #print(state)
            #print(state.shape)
            state = np.float32(state)
            # print('state___', state)
            if is_training and ep%2 == 0 and len(replay_buffer) > before_training*batch_size:
                action = policy_net.get_action(state)
            else:
                action = policy_net.get_action(state, exploitation=True)

            if not is_training:
                action = policy_net.get_action(state, exploitation=True)
            unnorm_action = np.array([action_unnormalized(action[0], ACTION_V_MAX, ACTION_V_MIN), action_unnormalized(action[1], ACTION_W_MAX, ACTION_W_MIN)])
            
            # ===== new reward ingredients ===== #
            
            next_state, original_reward, new_reward, done, goalbox, cord = env.step(unnorm_action, past_action)
            
            current_dist = state[SCAN_RANGE+1]
            temp_list.append([state, action, next_state, done, prev_cord, goalbox, [cord], current_dist])

            new_rewards += new_reward
            original_rewards += original_reward
            
            if goalbox:
                goal = True
            # print('action', unnorm_action,'r',reward)
            past_action = copy.deepcopy(action)

            rewards_current_episode += original_reward
            next_state = np.float32(next_state)
            if ep%2 == 0 or not len(replay_buffer) > before_training*batch_size:
                if goalbox:
                    print('***\n-------- Maximum Reward ----------\n****')
                    for _ in range(3):
                        replay_buffer.push(state, action, original_reward, next_state, done)
                else:
                    replay_buffer.push(state, action, original_reward, next_state, done)
            
            if len(replay_buffer) > before_training*batch_size and is_training and ep% 2 == 0:
                soft_q_loss, value_loss, policy_loss = soft_q_update(batch_size)
                print(soft_q_loss)
                print(value_loss)
                print(policy_loss)
                training_count += 1
            else:
                soft_q_loss, value_loss, policy_loss = 0, 0, 0
                
            state = copy.deepcopy(next_state)
            prev_cord = copy.deepcopy(cord)
            if done or step == MAX_STEPS-1 or goal:
                break
        
        if training_count > 0:
            ep_soft_q_loss.append(soft_q_loss / training_count)
            ep_value_loss.append(value_loss / training_count)
            ep_policy_loss.append(policy_loss / training_count)
        else:
            ep_soft_q_loss.append(0)
            ep_value_loss.append(0)
            ep_policy_loss.append(0)
            
        if temp_list[-1][3]:
            setHER(temp_list, 0.5, reward_mode, replay_buffer)
            setHER(temp_list, 1, reward_mode, replay_buffer)
            setHER(temp_list, 1.5, reward_mode, replay_buffer)
            
        if len(temp_list[:]) == MAX_STEPS:  # MAX_STEP 까지 충돌/성공 못한 경우
            setHER(temp_list, 0.5, reward_mode, replay_buffer)
            setHER(temp_list, 1, reward_mode, replay_buffer)
            setHER(temp_list, 1.5, reward_mode, replay_buffer)
        
        ep_new_rewards.append(new_rewards)
        ep_original_rewards.append(original_rewards)
        print('reward per ep: ' + str(rewards_current_episode))
        print('reward average per ep: ' + str(rewards_current_episode) + ' and break step: ' + str(step))
        if not ep%2 == 0:
            if len(replay_buffer) > before_training*batch_size:
                result = rewards_current_episode
                pub_result.publish(result)
        
        if ep%20 == 0:
            save_models(ep)
        if ep % 1000 == 0 and ep != 0:
            df1_name = "reward_ep" + str(ep) + " Ensemble_" + str(ensemble) + ".csv"
            df2_name = "new_rewad_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"
            df3_name = "soft_q_loss_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"
            df4_name = "value_loss_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"
            df5_name = "policy_loss_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"
            print("data saving...")
            df1 = pd.DataFrame(ep_original_rewards, columns=['reward'])
            df2 = pd.DataFrame(ep_new_rewards, columns=['reward'])
            df3 = pd.DataFrame(ep_soft_q_loss, columns=['loss'])
            df4 = pd.DataFrame(ep_value_loss, columns=['loss'])
            df5 = pd.DataFrame(ep_policy_loss, columns=['loss'])                
            # (22.07.28. park) 폴더 경로 자동으로 설정
            # (22.08.11. park) World별 Reward 따로 저장하도록 변경
            if not os.path.exists(dirPath + "/Rewards/" + world + '/SAC_Original Rewards/'):
                os.makedirs(dirPath + "/Rewards/" + world + '/SAC_Original Rewards/')
            if not os.path.exists(dirPath + "/Rewards/" + world + '/SAC_New Rewards/'):
                os.makedirs(dirPath + "/Rewards/" + world + '/SAC_New Rewards/')
            if not os.path.exists(dirPath + "/Loss/" + world + '/Soft Q Loss/'):
               os.makedirs(dirPath + "/Loss/" + world + '/Soft Q Loss/')
            if not os.path.exists(dirPath + "/Loss/" + world + '/Value Loss/'):
               os.makedirs(dirPath + "/Loss/" + world + '/Value Loss/')
            if not os.path.exists(dirPath + "/Loss/" + world + '/Policy Loss/'):
               os.makedirs(dirPath + "/Loss/" + world + '/Policy Loss/')
                
            df1.to_csv(dirPath + "/Rewards/" + world + '/SAC_Original Rewards/' + df1_name)
            df2.to_csv(dirPath + "/Rewards/" + world + '/SAC_New Rewards/' + df2_name)
            df3.to_csv(dirPath + "/Loss/" + world + '/Soft Q Loss/' + df3_name)
            df4.to_csv(dirPath + "/Loss/" + world + '/Value Loss/' + df4_name)
            df5.to_csv(dirPath + "/Loss/" + world + '/Policy Loss/' + df5_name)
            print("data saving completed..!")
