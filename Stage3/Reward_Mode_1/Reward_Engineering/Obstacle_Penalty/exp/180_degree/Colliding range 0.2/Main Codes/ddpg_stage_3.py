#!/home/park/anaconda3/envs/tbtorch/bin/python3
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
from sac_models import *
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

print("Catkin_ws2")
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
SCAN_RANGE = 10
STATE_DIMENSION = SCAN_RANGE + ACTION_DIMENSION + 4

if reward_mode == 1:
    print("******************* reward mode : ", reward_dict[1], " ********************")
if reward_mode == 2:
    print("******************* reward mode : ", reward_dict[2], " ********************")
if reward_mode == 3:
    print("******************* reward mode : ", reward_dict[3], " ********************")


# ---Directory Path---#
dirPath = os.path.dirname(os.path.realpath(__file__))


# ---Functions to make network updates---#

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class NoisyLinear(nn.Linear):
    def __init__(self, in_features, out_features, sigma_init=0.017, bias=True):
        super(NoisyLinear, self).__init__(in_features, out_features, bias=bias)
        # print("out : ", out_features, " in : ", in_features)
        self.sigma_weight = nn.Parameter(torch.full((out_features, in_features), sigma_init))
        # print(self.sigma_weight)
        self.register_buffer("epsilon_weight", torch.zeros(out_features, in_features))
        if bias:
            self.sigma_bias = nn.Parameter(torch.full((out_features,), sigma_init))
            self.register_buffer("epsilon_bias", torch.zeros(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(3 / self.in_features)
        self.weight.data.uniform_(-std, std)
        self.bias.data.uniform_(-std, std)

    def forward(self, input):
        self.epsilon_weight.normal_()
        bias = self.bias
        if bias is not None:
            self.epsilon_bias.normal_()
            bias = bias + self.sigma_bias * self.epsilon_bias.data
            # print("sigma : ", self.sigma.shape)
            # print("epslion_weight : ", self.epsilon_weight.data.shape)
        return F.linear(input, self.weight + self.sigma_weight * self.epsilon_weight.data, bias)


# ---Ornstein-Uhlenbeck Noise for action---#

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.2, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_noise(self, t=0):
        ou_state = self.evolve_state()
        decaying = float(float(t) / self.decay_period)
        self.sigma = max(self.sigma - (self.max_sigma - self.min_sigma) * min(1.0, decaying), self.min_sigma)
        return ou_state


# ---Critic--#

def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1. / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim

        self.fc1 = nn.Linear(state_dim, 250)
        self.fc1.weight.data = fanin_init(self.fc1.weight.data.size())

        self.fa1 = nn.Linear(action_dim, 250)
        self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())

        self.fca1 = nn.Linear(500, 500)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

        self.fca2 = nn.Linear(500, 1)
        self.fca2.weight.data.uniform_(-EPS, EPS)

    def forward(self, state, action):
        xs = torch.relu(self.fc1(state))
        xa = torch.relu(self.fa1(action))
        x = torch.cat((xs, xa), dim=1)
        x = torch.relu(self.fca1(x))
        vs = self.fca2(x)
        return vs


# ---Actor---#

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w):
        super(Actor, self).__init__()
        self.state_dim = state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w

        self.fa1 = nn.Linear(state_dim, 500)
        self.fa1.weight.data = fanin_init(self.fa1.weight.data.size())

        self.fa2 = nn.Linear(500, 500)
        self.fa2.weight.data = fanin_init(self.fa2.weight.data.size())

        self.fa3 = nn.Linear(500, action_dim)
        self.fa3.weight.data.uniform_(-EPS, EPS)

    def forward(self, state):
        x = torch.relu(self.fa1(state))
        x = torch.relu(self.fa2(x))
        action = self.fa3(x)
        # print("before actfun : ", "v_vel : ", action[0], " a_vel : ", action[0])
        if state.shape <= torch.Size([self.state_dim]):
            action[0] = torch.sigmoid(action[0]) * self.action_limit_v
            action[1] = torch.tanh(action[1]) * self.action_limit_w
        else:
            action[:, 0] = torch.sigmoid(action[:, 0]) * self.action_limit_v
            action[:, 1] = torch.tanh(action[:, 1]) * self.action_limit_w
        return action


class SAC(object):
    def __init__(self, num_inputs, action_space, args):
        self.gamma = args.gamma
        self.tau = args.tau
        self.alpha = args.alpha
        
        self.policy_type = args.policy
        self.target_update_interval = args.target_update_interval
        self.automatic_entropy_tuning = args.automatic_entropy_tuning
        
        self.device = torch.device("cuda" if args.cuda else "cpu")
        
        self.critic = QNetwork(num_inputs, action_space.space[0], args.hidden_size).to(device=self.device)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr)
        
        self.critic_target = QNetwork(num_inputs, action_space[0], args.hidden_size).to(device=self.device)
        hard_update(self.critic_target, self.critic)
        
        if self.policy_type == "Gaussian":
            if self.automatic_entropy_tuning is True:
                self.target_entropy = -torch.prod(torch.Tensor(action_space.shape).to(self.device)).item()
                self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
                self.alpha_optim = Adam([self.log_alpha], lr=args.lr)
            
            self.policy = GaussianPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
        else:
            self.alpha = 0
            self.automatic_entropy_tuning = False
            self.policy = DeterministicPolicy(num_inputs, action_space.shape[0], args.hidden_size, action_space).to(self.device)
            self.policy_optim = Adam(self.policy.parameters(), lr=args.lr)
    
    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        if evaluate is False:
            action, _, _ = self.policy.sample(state)
        else:
            _, _, action = self.policy.sample(state)
        return action.detach().cpu().numpy()[0]
        
    def update_parameters(self, memory, batch_size, updates):
        # Sample a batch from numpy
        state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(batch_size)
        
        state_batch = torch.FloatTensor(state_batch).to(self.device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
        action_batch = torch.FloatTensor(action_batch).to(self.device)
        reward_batch = torch.FloatTensor(reward_batch).to(self.device)
        mask_batch = torch.FloatTensor(mask_batch).to(self.device)
        
        with torch.no_grad():
            next_state_action, next_state_log_pi, _ = self.policy.sample(next_state_batch)
            qf1_next_target, qf2_next_target = self.critic_target(next_state_batch, next_state_action)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward_batch + mask_batch * self.gamma * min_qf_next_target
        qf1, qf2 = self.critic(state_batch, action_batch)
        qf1_loss = F.mse_loss(qf1, next_q_value)
        qf2_loss = F.mse_loss(qf2, next_q_value)
        qf_loss = qf1_loss + qf2_loss
        
        self.critic_optim.zero_grad()
        qf_loss.backward()
        self.critic_optim.step()
        
        pi, log_pi, _ = self.policy.sample(state_batch)
        
        qf1_pi, qf2_pi = self.critic(state_batch, pi)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        
        policy_loss = ((self.alpha * log_pi) - min_qf_pi).mean()
        
        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()
        
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            
            self.alpha = self.log_alpha.exp()
            alpha_tlogs = self.alpha.clone()
        else:
            alpha_loss = torch.tensor(0.).to(self.device)
            alpha_tlogs = torch.tensor(self.alpha)
        
        if updates % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
        
        return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha_tlogs.item()
    
# ---Memory Buffer---#

class MemoryBuffer:
    def __init__(self, size):
        self.buffer = deque(maxlen=size)
        self.maxSize = size
        self.len = 0

    def sample(self, count):
        batch = []
        count = min(count, self.len)
        batch = random.sample(self.buffer, count)

        s_array = np.float32([array[0] for array in batch])
        a_array = np.float32([array[1] for array in batch])
        r_array = np.float32([array[2] for array in batch])
        new_s_array = np.float32([array[3] for array in batch])

        return s_array, a_array, r_array, new_s_array

    def len(self):
        return self.len

    def add(self, s, a, r, new_s):
        transition = (s, a, r, new_s)
        self.len += 1
        if self.len > self.maxSize:
            self.len = self.maxSize
        self.buffer.append(transition)


# ---Where the train is made---#

class Trainer:

    def __init__(self, state_dim, action_dim, action_limit_v, action_limit_w, ram, ACTION_DIMENSION):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_limit_v = action_limit_v
        self.action_limit_w = action_limit_w
        # print('w',self.action_limit_w)
        self.ram = ram
        # self.iter = 0
        self.noise = OUNoise(ACTION_DIMENSION)

        self.actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_limit_v, self.action_limit_w)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LEARNING_RATE)

        self.critic = Critic(self.state_dim, self.action_dim)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), LEARNING_RATE)
        self.pub_qvalue = rospy.Publisher('qvalue', Float32, queue_size=5)
        self.qvalue = Float32()

        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

    def get_exploitation_action(self, state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()  # actor net 기반
        # print('actionploi', action)
        return action.data.numpy()

    def get_exploration_action(self, state):
        state = torch.from_numpy(state)
        action = self.actor.forward(state).detach()
        # print('noise', noise)
        new_action = action.data.numpy()  # + noise
        # print('action_no', new_action)
        # noise[0] = round(noise[0], 4) * self.action_limit_v / 2
        # noise[1] = round(noise[1], 4) *self.action_limit_w
        # new_action[0] = np.clip(new_action[0] + noise[0], 0., self.action_limit_v)
        # new_action[1] = np.clip(new_action[1] + noise[1], -self.action_limit_w, self.action_limit_w)
        return new_action

    def optimizer(self):
        s_sample, a_sample, r_sample, new_s_sample = ram.sample(BATCH_SIZE)

        s_sample = torch.from_numpy(s_sample)
        a_sample = torch.from_numpy(a_sample)
        r_sample = torch.from_numpy(r_sample)
        new_s_sample = torch.from_numpy(new_s_sample)

        # -------------- optimize critic

        a_target = self.target_actor.forward(new_s_sample).detach()
        next_value = torch.squeeze(self.target_critic.forward(new_s_sample, a_target).detach())
        # y_exp = r _ gamma*Q'(s', P'(s'))
        y_expected = r_sample + GAMMA * next_value
        # y_pred = Q(s,a)
        y_predicted = torch.squeeze(self.critic.forward(s_sample, a_sample))
        # -------Publisher of Vs------
        self.qvalue = y_predicted.detach()
        self.pub_qvalue.publish(torch.max(self.qvalue))
        # print(self.qvalue, torch.max(self.qvalue))
        # ----------------------------
        loss_critic = F.smooth_l1_loss(y_predicted, y_expected)

        self.critic_optimizer.zero_grad()
        loss_critic.backward()
        self.critic_optimizer.step()

        # ------------ optimize actor
        pred_a_sample = self.actor.forward(s_sample)
        loss_actor = -1 * torch.sum(self.critic.forward(s_sample, pred_a_sample))

        self.actor_optimizer.zero_grad()
        loss_actor.backward()

        self.actor_optimizer.step()
        # (22.08.15. park) loss 출력 깔끔히 나오도록 변경
        #print("==============================================")
        #print()
        #print("actor loss : ", loss_actor.detach().numpy().tolist(), " critic_loss : ", loss_critic.detach().numpy().tolist())
        #print()
        #print("==============================================")
        soft_update(self.target_actor, self.actor, TAU)
        soft_update(self.target_critic, self.critic, TAU)
        
        return loss_actor.detach().numpy().tolist(), loss_critic.detach().numpy().tolist()

    # (22.08.11. park) 실험 별, 에피소드 별 모델 저장하도록 변경
    def save_models(self, ensemble, episode_count):
        if not os.path.exists(dirPath + '/Models/' + world + '/' + str(ensemble) + '/'):
                os.makedirs(dirPath + '/Models/' + world + '/' + str(ensemble) + '/')
        torch.save(self.target_actor.state_dict(),
                   dirPath + '/Models/' + world + '/' + str(ensemble) + '/' + str(episode_count) + '_actor.pt')
        torch.save(self.target_critic.state_dict(),
                   dirPath + '/Models/' + world + '/' + str(ensemble) + '/' + str(episode_count) + '_critic.pt')
        print('****Models saved***')

    def load_models(self, ensemble, episode):
        self.actor.load_state_dict(torch.load(dirPath + '/Models/' + world + '/' + str(ensemble) + '/' + str(episode) + '_actor.pt'))
        self.critic.load_state_dict(torch.load(dirPath + '/Models/' + world + '/' + str(ensemble) + '/' + str(episode) + '_critic.pt'))
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)
        print('***Models load***')


# ---Mish Activation Function---#
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
    return x * (torch.tanh(F.softplus(x)))


def newReward(prev_cord, cord, step, reward_mode):  # for HER
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

	# (22.07.22. kwon) reward mode 에 상관없이 아래 조건문이 사용되므로 조건문 밖에서 한 번만 실행하는 것으로 변경
    if current_distance < 0.15:  # 목적지에 도달했을 시 500점 추가, HER을 적용하기 위해
		# (22.07.20. kwon) 실제 보상과 일치시키기 위해 reward += 500 으로 수정
		# (22.07.22. kwon) 위 사항 재수정. immediate reward 이므로 += 에서 = 으로 수정
        reward = goal_reward  # 충돌 직전의 좌표를 목적지로 생각하기 때문에, 충돌 리워드는 없음
        newgoal_box = True

    goal_angle = math.atan2(goal_y - current_y, goal_x - current_x)
    new_goal_angle = math.atan2(newgoal_y - current_y, newgoal_x - current_x)

    return reward, current_distance, new_goal_angle - goal_angle, newgoal_box


# (22.07.20 kwon) setHER 함수 주석 추가
# temp_list : HER 설정을 위한 list
# step : 과거 몇 step 까지 HER 에 사용할지를 결정하는 변수
def setHER(temp_list, step, reward_mode):
    # (22.07.20 kwon) 조건문 주석 추가
    # temp_list[-1][3] : 충돌인 경우 확인 (충돌 = True)
    # len(temp_list[:]) > step   :  (전체스텝 - step) 에 대해 HER 적용이 가능한지 확인
    # if temp_list[-1][3] and len(temp_list[:]) > step:
    print("modified goal data saving...")
    # (22.07.20. kwon) new_goal 주석 추가
    new_goal_x = temp_list[-step][6][0][0]  # 전체스텝 - step 의 위치를 new_goal 로 설정
    new_goal_y = temp_list[-step][6][0][1]
    print("new_goal : " + str(new_goal_x) + ", " + str(new_goal_y))
    for i in range(len(temp_list[:]) - step):  # 전체스텝 - step 만큼 반복
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

        new_reward, current_distance, diff_goal_angle, n_goal_box = newReward(prev_cord, cd, i, reward_mode)
        st[-1] = current_distance
        st[-2] = diff_goal_angle + st[-2]

        _, n_current_distance, n_diff_goal_angle, _ = newReward(temp_list[i + 1][4], n_cd, i + 1, reward_mode)
        n_st[-1] = n_current_distance
        n_st[-2] = n_diff_goal_angle + n_st[-2]

        if n_goal_box:
            print('setHER goal step : ' + str(i))  # (22.07.22 kwon) HER 제대로 되었는지 확인을 위함
            for k in range(3):
                ram.add(st, act, new_reward, n_st)
            break
        else:
            ram.add(st, act, new_reward, n_st)
        del cd, n_cd


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
# ram = MemoryBuffer(MAX_BUFFER)
# trainer = Trainer(STATE_DIMENSION, ACTION_DIMENSION, ACTION_V_MAX, ACTION_W_MAX, ram)

noise = OUNoise(ACTION_DIMENSION)
reward_list = []
new_reward_list = []
# trainer.load_models(140)

if __name__ == '__main__':
    rospy.init_node('ddpg_stage_3')
    pub_result = rospy.Publisher('result', Float32, queue_size=5)
    result = Float32()
    before_training = 1
    env = Env(action_dim=ACTION_DIMENSION)
    goal = False
   # for ensemble in range(10):
    actor_loss = []
    critic_loss = []
    reward_list = []
    new_reward_list = []
    ram = MemoryBuffer(MAX_BUFFER)
    past_action = np.zeros(ACTION_DIMENSION)
    trainer = Trainer(STATE_DIMENSION, ACTION_DIMENSION, ACTION_V_MAX, ACTION_W_MAX, ram, ACTION_DIMENSION)
    for ep in range(MAX_EPISODES):
        ep_actor_loss = 0
        ep_critic_loss = 0
        training_count = 0
        done = False
        if not goal:
            state, prev_cord = env.reset()
        goal = False

        if is_training and ram.len >= before_training * MAX_STEPS * 4:
            print('---------------------------------')
            print('Episode: ' + str(ep) + ' training')
            print('---------------------------------')
        else:
            print('---------------------------------')
            print('Episode: ' + str(ep) + ' adding to memory')
            print('---------------------------------')

        original_rewards = 0.
        new_rewards = 0.
        temp_list = []

        for step in range(MAX_STEPS):
            state = np.float32(state)
            # print("heading : ", state[-2])

            if is_training and ram.len >= before_training * MAX_STEPS * 4:
                action = trainer.get_exploration_action(state)
                N = copy.deepcopy(noise.get_noise(t=step / (1 + 0.01 * ep)))
                N[0] = round(N[0], 4) * ACTION_V_MAX / 2
                N[1] = round(N[1], 4) * ACTION_W_MAX
                action[0] = np.clip(action[0] + N[0], 0., ACTION_V_MAX)
                action[1] = np.clip(action[1] + N[1], -ACTION_W_MAX, ACTION_W_MAX)
                # (22.08.15. park)_action 출력 깔끔하게 나오도록 변경
                #print("==============================================")
                #print()
                #print("exploration action : ", action)
                #print()
                #print("==============================================")
            else:
                action = trainer.get_exploration_action(state)
                #print("==============================================")
                #print()
                #print("exploitation action : ", action)
                #print()
                #print("==============================================")
            if not is_training:
                action = trainer.get_exploitation_action(state)
            
            #action[1] = ACTION_V_MAX * 0.7
            #print("angular : ", action[1])

            # (22.07.21 park) Reward 명 변경
            next_state, original_reward, new_reward, done, goalbox, cord = env.step(action, past_action)

            # print('action', action,'r',reward)
            past_action = action

            # ===== new reward ingredients ===== #
            temp_list.append([state, action, next_state, done, prev_cord, goalbox, [cord]])
            
            new_rewards += new_reward
            original_rewards += original_reward
            next_state = np.float32(next_state)

            # (22.08.15. park) Reward 확인용
            #print("==============================================")
            #print()
            #print("New reward : ", new_rewards)
            #print()
            #print("==============================================")
            
             # (22.07.22. kwon) reward_mode 별 ram 에 추가해야하는 immediate reward 로 list 생성 -
            reward_arg = [original_reward, new_reward, new_rewards]

            if goalbox:
                goal = True
                print('***\n-------- Maximum Reward ----------\n****')
                # (22.07.21. park) Reward mode 부분 조건문으로 추가
                # (22.07.22. kwon) 조건문 사용 안 하도록 코드 최적화
                for _ in range(3):
                    ram.add(state, action, reward_arg[reward_mode - 1], next_state)

            else:
                # (22.07.21. park) Reward mode 부분 조건문으로 추가
                # (22.07.22. kwon) 조건문 사용 안 하도록 코드 최적화
                ram.add(state, action, reward_arg[reward_mode - 1], next_state)

            state = copy.deepcopy(next_state)
            prev_cord = copy.deepcopy(cord)
            
            #print("memory length: ", ram.len)
            if ram.len >= MAX_STEPS * 4 and is_training:
                # var_v = max([var_v*0.99999, 0.005*ACTION_V_MAX])
                # var_w = max([var_w*0.99999, 0.01*ACTION_W_MAX])
                # for epoch in range(1):
                #print("training..........")
                actor_l, critic_l = trainer.optimizer()
                ep_actor_loss += actor_l
                ep_critic_loss += critic_l
                training_count += 1
                #print("actor loss: ", actor_l, type(actor_l))
                #print("critic loss: ", critic_l, type(critic_l))
            
            
            if done or step == MAX_STEPS - 1 or goal:
                print('Original_reward per ep: ' + str(original_rewards))
                print('*\nbreak step: ' + str(step) + '\n*')
                result = original_rewards
                pub_result.publish(result)
                break

        if training_count > 0:
            actor_loss.append(ep_actor_loss / training_count)
            critic_loss.append(ep_critic_loss / training_count)
        else:
            actor_loss.append(0)
            critic_loss.append(0)

        if temp_list[-1][3] and len(temp_list[:]) > 50:
            setHER(temp_list, 5, reward_mode)
            setHER(temp_list, 25, reward_mode)
            setHER(temp_list, 50, reward_mode)
            #setHER(temp_list, round(len(temp_list)*0.25), reward_mode)
            #setHER(temp_list, round(len(temp_list)*0.5), reward_mode)
            #setHER(temp_list, round(len(temp_list)*0.75), reward_mode)

        if len(temp_list[:]) == MAX_STEPS:  # MAX_STEP 까지 충돌/성공 못한 경우
            setHER(temp_list, 50, reward_mode)
            setHER(temp_list, 150, reward_mode)
            setHER(temp_list, 250, reward_mode)
            #setHER(temp_list, round(len(temp_list)*0.25), reward_mode)
            #setHER(temp_list, round(len(temp_list)*0.5), reward_mode)
            #setHER(temp_list, round(len(temp_list)*0.75), reward_mode)

        print("saving completed!")
        print("# of data : ", ram.len)
        reward_list.append(original_rewards)
        new_reward_list.append(new_rewards)
        if ep % 20 == 0:
            trainer.save_models(ensemble, ep)
        if ep % 1000 == 0 and ep > 0:
            df1_name = "reward_ep" + str(ep) + " Ensemble_" + str(ensemble) + ".csv"
            df2_name = "new_rewad_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"
            df3_name = "actor_loss_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"
            df4_name = "critic_loss_ep" + str(ep) + " Ensemble " + str(ensemble) + ".csv"

            print("data saving...")
            df1 = pd.DataFrame(reward_list, columns=['reward'])
            df2 = pd.DataFrame(new_reward_list, columns=['reward'])
            df3 = pd.DataFrame(actor_loss, columns=['loss'])
            df4 = pd.DataFrame(critic_loss, columns=['loss'])
                            
            # (22.07.28. park) 폴더 경로 자동으로 설정
            # (22.08.11. park) World별 Reward 따로 저장하도록 변경
            if not os.path.exists(dirPath + "/Rewards/" + world + '/Original Rewards/'):
                os.makedirs(dirPath + "/Rewards/" + world + '/Original Rewards/')
            if not os.path.exists(dirPath + "/Rewards/" + world + '/New Rewards/'):
                os.makedirs(dirPath + "/Rewards/" + world + '/New Rewards/')
            if not os.path.exists(dirPath + "/Loss/" + world + '/Critic Loss/'):
                os.makedirs(dirPath + "/Loss/" + world + '/Critic Loss/')
            if not os.path.exists(dirPath + "/Loss/" + world + '/Actor Loss/'):
                os.makedirs(dirPath + "/Loss/" + world + '/Actor Loss/')
                
            df1.to_csv(dirPath + "/Rewards/" + world + '/Original Rewards/' + df1_name)
            df2.to_csv(dirPath + "/Rewards/" + world + '/New Rewards/' + df2_name)
            df3.to_csv(dirPath + "/Loss/" + world + '/Actor Loss/' + df3_name)
            df4.to_csv(dirPath + "/Loss/" + world + '/Critic Loss/' + df4_name)

            print("data saving completed!!")
    del ram

print('Completed Training')
print('processing time : ', time.time() - start_time)
