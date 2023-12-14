from dqn_Agent import *
import pickle
from Environment import *
from DDPG import DDPG_CA
from sb3_contrib import ARS
from stable_baselines3 import PPO


diretorio_agentes = "Agentes/"
with open("dataframe2.pkl", 'rb') as file:
    df4 = pickle.load(file)
test = df4.iloc[500:700]
test = test.reset_index()

env =  env = InvOptEnv_unico_produto(500, 500, test, 500)
envdqn = InvOptEnv_unico_produto_dqn(500, 500, test, 500)
device = "cpu"
resultados = {}

modeldqn = QNetwork(state_size=8,action_size=50,seed=0)
modeldqn.load_state_dict(torch.load(diretorio_agentes+'agentDQN.pt'))
modeldqn.eval()

agentddpg = DDPG_CA(8, 1, 500)
agentddpg.load()

model_ars = ARS.load("Agentes/agentARS")
model_ppo = PPO.load("Agentes/agentPPO")

def test_model_dqn(model):
    envdqn.reset()
    profit = []
    actions = []
    invs = []
    backlogs = []
    done = False
    state = envdqn.state
    i = 0
    while i<100:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = modeldqn(state)
        action = np.argmax(action_values.cpu().data.numpy())
        action = int(action*100)
        actions.append(action)
        next_state, reward, done, _ = envdqn.step(action)
        state = next_state
        invs.append(envdqn.Nivel_estoque)
        backlogs.append(envdqn.backlog)
        profit.append(reward)
        i+=1
    
    return profit, actions, backlogs, invs

def test_model_ddpg(agent):
    state = envdqn.reset()
    profit = []
    actions = []
    invs = []
    backlogs = []
    done = False
    i = 0
    while i<100:
        action = agent.select_action(state)
        action = int((action).clip(envdqn.action_space.low, envdqn.action_space.high)[0])
        next_state, reward, done, info = envdqn.step(np.float32(action))
        state = next_state
        actions.append(action)
        invs.append(envdqn.Nivel_estoque)
        backlogs.append(envdqn.backlog)
        profit.append(reward)
        i+=1
           
    return profit, actions, backlogs, invs

def test_model_ars(model):
    env.reset()
    profit = []
    actions = []
    invs = []
    backlogs = []
    done = False
    state = env.state
    i = 0
    while i<100:
        action, _states = model.predict(state.astype(np.float32), deterministic=True)
        action = int(action[0])
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        invs.append(env.Nivel_estoque)
        backlogs.append(env.backlog)
        profit.append(reward)
        i+=1
    
    return profit, actions, backlogs, invs

def test_model_ppo(model):
    env.reset()
    profit = []
    actions = []
    invs = []
    backlogs = []
    done = False
    state = env.state
    i = 0
    while i<100:
        action, _states = model.predict(state.astype(np.float32), deterministic=True)
        action = int(action[0])
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        invs.append(env.Nivel_estoque)
        backlogs.append(env.backlog)
        profit.append(reward)
        i+=1
    
    return profit, actions, backlogs, invs

def test_policy_sS():
    env.reset()
    profit = []
    actions = []
    invs = []
    backlogs = []
    done = False
    state = env.state
    i = 0
    while i<100:
        if env.Nivel_estoque <= 350:
            if env.Em_entrega == 0:
                action = 400
            else:
                action = 0
        else:
            action = 0
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        invs.append(env.Nivel_estoque)
        backlogs.append(env.backlog)
        profit.append(reward)
        i+=1
    
    return profit, actions, backlogs, invs

profit, actions, backlogs, invs = test_model_dqn(modeldqn)
resultados["DQN"] = {"Receita": profit,
                     "Acoes": actions,
                     "backlog": backlogs,
                     "estoque": invs}

profit, actions, backlogs, invs = test_model_ddpg(agentddpg)
resultados["DDPG"] = {"Receita": profit,
                     "Acoes": actions,
                     "backlog": backlogs,
                     "estoque": invs}

profit, actions, backlogs, invs = test_model_ars(model_ars)
resultados["ARS"] = {"Receita": profit,
                     "Acoes": actions,
                     "backlog": backlogs,
                     "estoque": invs}

profit, actions, backlogs, invs = test_model_ppo(model_ppo)
resultados["PPO"] = {"Receita": profit,
                     "Acoes": actions,
                     "backlog": backlogs,
                     "estoque": invs}

profit, actions, backlogs, invs = test_policy_sS()
resultados["sS"] = {"Receita": profit,
                     "Acoes": actions,
                     "backlog": backlogs,
                     "estoque": invs}

print(np.sum(resultados['sS']['Receita']))
print(np.sum(resultados['DDPG']['Receita']))
print(np.sum(resultados['ARS']['Receita']))

print(len([x for x in resultados['sS']['Acoes'] if x >= 1]))
print(len([x for x in resultados['DDPG']['Acoes'] if x >= 1]))
print(len([x for x in resultados['ARS']['Acoes'] if x >= 1]))


import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# Suas quatro listas de dados
dqn_profit = np.cumsum(resultados['DQN']['Receita'])
ddpg_profit = np.cumsum(resultados['DDPG']['Receita'])
ars_profit = np.cumsum(resultados['ARS']['Receita'])
ppo_profit = np.cumsum(resultados['PPO']['Receita'])
sS_profit = np.cumsum(resultados['sS']['Receita'])
#dqn_profit = resultados['DQN']['Acoes']
#ddpg_profit = resultados['DDPG']['Acoes']
#ars_prodit = resultados['ARS']['Acoes']
#sS_profit = resultados['sS']['Acoes']
print(len(dqn_profit))
# Criar um DataFrame do pandas
data = pd.DataFrame({'X': [i for i in range(1,101)], 'DQN': dqn_profit, 'DDPG': ddpg_profit,
                      'ARS': ars_profit,'PPO': ppo_profit, 'sS': sS_profit})

# Configurar o estilo do seaborn (opcional)
sns.set(style="whitegrid")

# Criar o gráfico de linhas usando Seaborn
sns.lineplot(data=data, x='X', y='DQN', label='DQN')
sns.lineplot(data=data, x='X', y='DDPG', label='DDPG')
sns.lineplot(data=data, x='X', y='ARS', label='ARS')
sns.lineplot(data=data, x='X', y='sS', label='sS')
sns.lineplot(data=data, x='X', y='PPO', label='PPO')

# Adicionar rótulos e título
plt.xlabel('Dia')
plt.ylabel('Receita')
plt.title('Gráfico de Linhas')

# Adicionar legenda
plt.legend()

# Mostrar o gráfico
plt.show()