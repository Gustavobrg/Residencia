from sb3_contrib import ARS
import pickle
from env_v2 import *
import numpy as np
import matplotlib.pyplot as plt



with open("dataframe3.pkl", 'rb') as file:
    df4 = pickle.load(file)
prices = [1.58,1.68,1.68,1.58,1.68,4.98,1.5,1.68,1.68,1.68,1.58,1.5,1.5,1.48,1.6,1.68,1.68,1.68,1.6]
    
env = InvOptEnv_produtos([1000]*19, [1000]*19, df4, prices, 400, 19)

total_timesteps = 1000
num_episodes = 1000

from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback

eval_callback = EvalCallback(env, best_model_save_path='./logs/',
                             log_path='./logs/', eval_freq=8116,
                             deterministic=True, render=False)

# Policy can be LinearPolicy or MlpPolicy
model = ARS('MlpPolicy', env, verbose=1)
model.learn(total_timesteps=8116000, callback=eval_callback, log_interval=1)
model.save("Agentes/agentARS")