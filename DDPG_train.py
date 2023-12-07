import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from itertools import count
import torch.optim as optim
import math
from Environment import *
import pickle
import matplotlib.pyplot as plt



with open("dataframe2.pkl", 'rb') as file:
    df4 = pickle.load(file)

env = InvOptEnv_unico_produto_dqn(500, 500, df4, 500)

max_action = 500
state_dim = 8
action_dim = 1
capacity=1000000
batch_size=64
update_iteration=20
tau=0.001 # tau for soft updating
gamma=0.99 # discount factor
directory = "Agentes/"
hidden1=20 # hidden layer for actor
hidden2=64. #hiiden laye for critic
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)

class Replay_buffer():
    '''
    Code based on:
    https://github.com/openai/baselines/blob/master/baselines/deepq/replay_buffer.py
    Expects tuples of (state, next_state, action, reward, done)
    '''
    def __init__(self, max_size=10000):
        self.storage = []
        self.max_size = max_size
        self.ptr = 0

    def push(self, data):
        if len(self.storage) == self.max_size:
            self.storage[int(self.ptr)] = data
            self.ptr = (self.ptr + 1) % self.max_size
        else:
            self.storage.append(data)

    def sample(self, batch_size):
        ind = np.random.randint(0, len(self.storage), size=batch_size)
        x, y, u, r, d = [], [], [], [], []

        for i in ind:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.array(R, copy=False))
            d.append(np.array(D, copy=False))

        return np.array(x), np.array(y), np.array(u), np.array(r).reshape(-1, 1), np.array(d).reshape(-1, 1)
    

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 64)
        self.l2 = nn.Linear(64, 32)
        self.l3 = nn.Linear(32, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))

        # Mapeia a saída diretamente para [0, 1]
        x = torch.sigmoid(self.l3(x))
        
        # Escala para o intervalo [0, max_action]
        x = self.max_action * x
        return x


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()

        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.l2 = nn.Linear(256 , 128)
        self.l3 = nn.Linear(128, 1)

    def forward(self, x, u):
        x = F.relu(self.l1(torch.cat([x, u], 1)))
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x
    

class DDPG(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)

        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)
        self.replay_buffer = Replay_buffer()
        self.writer = SummaryWriter(directory)

        self.num_critic_update_iteration = 0
        self.num_actor_update_iteration = 0
        self.num_training = 0

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def update(self):

        for it in range(update_iteration):
            # Sample replay buffer
            x, y, u, r, d = self.replay_buffer.sample(batch_size)
            state = torch.FloatTensor(x).to(device)
            action = torch.FloatTensor(u).to(device)
            next_state = torch.FloatTensor(y).to(device)
            done = torch.FloatTensor(1-d).to(device)
            reward = torch.FloatTensor(r).to(device)

            # Compute the target Q value
            target_Q = self.critic_target(next_state, self.actor_target(next_state))
            target_Q = reward + (done * gamma * target_Q).detach()

            # Get current Q estimate
            current_Q = self.critic(state, action)

            # Compute critic loss
            critic_loss = F.mse_loss(current_Q, target_Q)
            self.writer.add_scalar('Loss/critic_loss', critic_loss, global_step=self.num_critic_update_iteration)
            # Optimize the critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            # Compute actor loss
            actor_loss = -self.critic(state, self.actor(state)).mean()
            self.writer.add_scalar('Loss/actor_loss', actor_loss, global_step=self.num_actor_update_iteration)

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            self.num_actor_update_iteration += 1
            self.num_critic_update_iteration += 1

    def save(self):
        torch.save(self.actor.state_dict(), directory + 'actorDDPG.pth')
        torch.save(self.critic.state_dict(), directory + 'criticDDPG.pth')
        # print("====================================")
        # print("Model has been saved...")
        # print("====================================")

    def load(self):
        self.actor.load_state_dict(torch.load(directory + 'actorDDPG.pth'))
        self.critic.load_state_dict(torch.load(directory + 'criticDDPG.pth'))
        print("====================================")
        print("model has been loaded...")
        print("====================================")


def run(modo="train", carregar=False, episodies=1000, exploration_noise=100):
    scores = []
    agent = DDPG(state_dim, action_dim, max_action)
    ep_r = 0
    if modo == 'test':
        agent.load()
        for i in range(1):
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                action = (action).clip(env.action_space.low, env.action_space.high)[0]
                print(action)
                next_state, reward, done, info = env.step(np.float32(action))
                print(action)
                print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}, action is \t{}".format(i, reward, t, action))
                ep_r += reward
                if done or t >= 100:
                    print("Ep_i \t{}, the ep_r is \t{:0.2f}, the step is \t{}, action is \t{}".format(i, ep_r, t, action))
                    ep_r = 0
                    break
                state = next_state

    elif modo == 'train':
        if carregar: agent.load()
        total_step = 0
        for i in range(episodies):
            total_reward = 0
            step =0
            state = env.reset()
            for t in count():
                action = agent.select_action(state)
                #print(action)
                #if action < 1:
                  #action = [0]
                
                #else:
                action = (action + np.random.normal(0, exploration_noise, size=env.action_space.shape[0])).clip(
                env.action_space.low, env.action_space.high)
                
                #print(action)
                #print("===================")                

                next_state, reward, done, _ = env.step(int(action[0]))
                agent.replay_buffer.push((state, next_state, action, reward, np.float32(done)))

                state = next_state
                if done:
                    break
                step += 1
                total_reward += reward
            total_step += step+1
            print("Total T:{} Episode: \t{} Total Reward: \t{:0.2f}".format(total_step, i, total_reward))
            agent.update()
           # "Total T: %d Episode Num: %d Episode T: %d Reward: %f

            if i % 30 == 0:
                agent.save()
            
            scores.append(total_reward)
    else:
        raise NameError("mode wrong!!!")
    
    return scores

scores = run()

plt.plot(np.arange(len(scores)),scores)
plt.ylabel('Reward')
plt.xlabel('Epsiode #')
plt.show()