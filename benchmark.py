from dqn_Agent import *
import pickle
from Environment import *

model = QNetwork(state_size=13,action_size=100,seed=0)
model.load_state_dict(torch.load('agent2.pt'))
model.eval()

with open("dataframe.pkl", 'rb') as file:
    df4 = pickle.load(file)


test = df4.loc[df4['item_nbr']=='1047679'].iloc[300:500]
test = test.reset_index()


ordering_cost = 1
holding_cost = 0.01
penalty = 100
fixed_cost = 80
env =  InvOptEnv_unico_produto(9000, 7000, ordering_cost, holding_cost, penalty, fixed_cost, test, 100)
device = "cpu"

def test_model(model):
    env.reset()
    profit = 0
    actions = []
    invs = []
    backlogs = []
    done = False
    state = env.state
    while not done:
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action_values = model(state)
        action = np.argmax(action_values.cpu().data.numpy())
        action = action*100
        actions.append(action)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        invs.append(env.Nivel_estoque)
        backlogs.append(env.backlog)
        profit += reward
    
    return profit, actions, backlogs, invs


profit, actions, backlogs, invs = test_model(model)
print(profit)
print(actions)
print(backlogs)
print(invs)