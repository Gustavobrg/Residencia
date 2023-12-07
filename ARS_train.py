from sb3_contrib import ARS
import pickle
from Environment import *
import numpy as np
import matplotlib.pyplot as plt



with open("dataframe2.pkl", 'rb') as file:
    df4 = pickle.load(file)

    
env = InvOptEnv_unico_produto(500, 500, df4, 500)

total_timesteps = 1000
num_episodes = 1000

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=8116,
                             deterministic=True, render=False)

# Policy can be LinearPolicy or MlpPolicy
model = ARS("LinearPolicy", env, verbose=1)
model.learn(total_timesteps=8116000, callback=eval_callback, log_interval=1)
model.save("Agentes/agentARS")