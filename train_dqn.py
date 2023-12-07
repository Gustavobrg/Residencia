import pickle
from Environment import *

from dqn_Agent import *
import numpy as np
import matplotlib.pyplot as plt


with open("dataframe2.pkl", 'rb') as file:
    df4 = pickle.load(file)

df4 = df4[1000:1940].reset_index()
    
env = InvOptEnv_unico_produto_dqn(500, 500, df4, 400)


agent = Agent(state_size=8,action_size=50,seed=0)

def dqn(env, n_episodes= 1000, max_t = 10000, eps_start=1.0, eps_end = 0.01,
       eps_decay=0.995):
    """Deep Q-Learning

    Params
    ======
        n_episodes (int): maximum number of training epsiodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): mutiplicative factor (per episode) for decreasing epsilon

    """
    scores = [] # list containing score from each episode
    eps = eps_start
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        state = np.array(state)
        score = 0
        for t in range(max_t):
            action = agent.act(state,eps)
            action = action*10
            next_state,reward,done, _ = env.step(action)
            next_state = np.array(next_state)
            agent.step(state,action/100,reward,next_state,done)
            ## above step decides whether we will train(learn) the network
            ## actor (local_qnetwork) or we will fill the replay buffer
            ## if len replay buffer is equal to the batch size then we will
            ## train the network or otherwise we will add experience tuple in our
            ## replay buffer.
            state = next_state
            score += reward
            if done:
                print('episode'+str(i_episode)+':', score)
                scores.append(score)
                break
        eps = max(eps*eps_decay,eps_end)## decrease the epsilon
    return scores

scores = dqn(env)
torch.save(agent.qnetwork_local.state_dict(), "Agentes/agentDQN.pt")


plt.plot(np.arange(len(scores)),scores)
plt.ylabel('Reward')
plt.xlabel('Epsiode #')
plt.show()