import numpy as np

arquivo_npz = np.load('logs\evaluations.npz')

print(arquivo_npz['results'])
print(arquivo_npz['timesteps'])