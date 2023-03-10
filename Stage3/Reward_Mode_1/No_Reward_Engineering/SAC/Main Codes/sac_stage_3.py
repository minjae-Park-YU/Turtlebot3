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

#****************************
is_training = True

#load_models(120)   
hard_update(target_value_net, value_net)
max_episodes  = 1001
max_steps   = 300
rewards     = []
batch_size  = 256



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

        for step in range(max_steps):
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

            next_state, original_reward, new_reward, done, goalbox, cord = env.step(unnorm_action, past_action)
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
                soft_q_update(batch_size)
            state = copy.deepcopy(next_state)
            
            if done or step == max_steps-1 or goal:
                break
        
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
        if ep % 1000 == 0:
            df1_name = "reward_ep" + str(ep) + " Ensemble_" + str(ensemble) + ".csv"
            df2_name = "new_rewad_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"
            #df3_name = "actor_loss_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"
            #df4_name = "critic_loss_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"

            print("data saving...")
            df1 = pd.DataFrame(ep_original_rewards, columns=['reward'])
            df2 = pd.DataFrame(ep_new_rewards, columns=['reward'])
            #df3 = pd.DataFrame(actor_loss, columns=['loss'])
            #df4 = pd.DataFrame(critic_loss, columns=['loss'])
                            
            # (22.07.28. park) 폴더 경로 자동으로 설정
            # (22.08.11. park) World별 Reward 따로 저장하도록 변경
            if not os.path.exists(dirPath + "/Rewards/" + world + '/SAC_Original Rewards/'):
                os.makedirs(dirPath + "/Rewards/" + world + '/SAC_Original Rewards/')
            if not os.path.exists(dirPath + "/Rewards/" + world + '/SAC_New Rewards/'):
                os.makedirs(dirPath + "/Rewards/" + world + '/SAC_New Rewards/')
            #if not os.path.exists(dirPath + "/Loss/" + world + '/Critic Loss/'):
            #   os.makedirs(dirPath + "/Loss/" + world + '/Critic Loss/')
            #if not os.path.exists(dirPath + "/Loss/" + world + '/Actor Loss/'):
            #   os.makedirs(dirPath + "/Loss/" + world + '/Actor Loss/')
                
            df1.to_csv(dirPath + "/Rewards/" + world + '/SAC_Original Rewards/' + df1_name)
            df2.to_csv(dirPath + "/Rewards/" + world + '/SAC_New Rewards/' + df2_name)
            #df3.to_csv(dirPath + "/Loss/" + world + '/Actor Loss/' + df3_name)
            #df4.to_csv(dirPath + "/Loss/" + world + '/Critic Loss/' + df4_name)
            print("data saving completed..!")
