import gym
from gym import spaces
from gym.utils import seeding
import random
import numpy as np

prices = [1.58,
 1.68,
 1.68,
 1.58,
 1.68,
 4.98,
 1.5,
 1.68,
 1.68,
 1.68,
 1.58,
 1.5,
 1.5,
 1.48,
 1.6,
 1.68,
 1.68,
 1.68,
 1.6]

class InvOptEnv_produtos(gym.Env):
    def __init__(self,
                 max_estoque,
                 estoque_inicial,
                 time_serie,
                 price,
                 horizonte,
                 num_produtos):
      
      self.num_produtos = num_produtos
      self.estoque_inicial = estoque_inicial
      self.custoproduto = np.array(price)
      self.Co = self.custoproduto*0.33   # Custo Pedido
      self.Ch = self.custoproduto*0.07    # Custo de manter estoque
      self.Cp = 50     # penalidade por ter demanda e não ter estoque
      self.custo_fixo_pedido = 30
      self.score = 0

      self.action_space = spaces.Box(low=0, high=8000, shape=(num_produtos,), dtype=int)   # -> ações discretas
      self.observation_space = spaces.Box(
      low=0, high=np.inf, shape=(num_produtos*8,), dtype=np.float16)

      # Initialize the time-series parameters

      self.demand_ts = time_serie

      self.max_estoque = max_estoque   # -> Maximum inventory level

      self.reset()   # reset the state vector
      self.seed()

      self.horizonte = horizonte

    def seed(self, seed=None):
      self.np_random, seed = seeding.np_random(seed)
      return [seed]

    def reset(self):
      # Resetar o estado
      self.Nivel_estoque = np.array(self.estoque_inicial)
      self.Transicao = np.zeros(self.num_produtos, dtype=int)
      self.Demanda = np.zeros(self.num_produtos, dtype=int)
      self.Pedido = np.zeros(self.num_produtos, dtype=int)
      self.Em_entrega = np.zeros(self.num_produtos, dtype=int)
      self.Distancia_feriado = np.zeros(self.num_produtos, dtype=int)
      self.Dia_semana = np.zeros(self.num_produtos, dtype=int)
      self.historico = np.zeros((self.num_produtos, 7), dtype=int)
      self.state = self._get_observation()

      # Resetar o tempo de entrega
      self.Tempo_entrega = np.random.randint(5, 8, size=self.num_produtos)

      # Resetar variáveis usadas
      self.backlog = np.zeros(self.num_produtos, dtype=int)
      self.lista_em_entrega = [[0] for _ in range(self.num_produtos)]
      self.produtos_a_mais = np.zeros(self.num_produtos, dtype=int)

      # Resetar o período
      self.periodo = 8

      return self.state

    def _get_tempo_entrega(self):
        return np.random.randint(5, 7, self.num_produtos).tolist()

    def _get_demanda(self):
      return self.demand_ts.iloc[self.periodo, :self.num_produtos].values.tolist()

    #obtem a demanda do dia atual
    def _get_diasferiado(self):
        return [self.demand_ts.loc[self.periodo, 'dias_ate_proximo_feriado']]*self.num_produtos

    #obtem a demanda do dia atual
    def _get_dia_da_semana(self):
        return [self.demand_ts.loc[self.periodo, 'dia_da_semana']]*self.num_produtos

    #obtem a demanda do dia atual
    def _get_historico(self):
        indices = np.arange(self.periodo - 7 - 1, self.periodo-1)
        historico = self.demand_ts.iloc[indices, :self.num_produtos].values.T
        return historico

    def _OnOrder_update(self, tempo_entrega, action, listas):
        nova_lista = []
        for lista, tempo, acao in zip(listas, tempo_entrega, action):
            if len(lista) > tempo:
                lista[tempo] += acao
            else:
                pad_length = max(0, tempo - len(lista) + 1)
                zero_padding = [0] * pad_length
                lista.extend(zero_padding)
                lista[tempo] += acao
            nova_lista.append(lista)

        self.lista_em_entrega = nova_lista

    #obtem estado atual
    def _get_observation(self):
          historico_obs = [sublista[0] for sublista in self.historico]
          media = [int(valor) for valor in np.mean(self.historico, axis=1)]
          obs = np.nan_to_num([self.Nivel_estoque, self.Transicao, self.Pedido, self.Em_entrega, self.Distancia_feriado, self.Dia_semana, historico_obs, media])
          obs = obs.T
          return obs.flatten()

    def _calcular_custo(self):
        custo_fixo_pedido = np.where(self.Pedido != 0, self.custo_fixo_pedido, 0)
        
        reward = (
            np.array(self.Demanda) * np.array(self.custoproduto) -
            np.array(self.Ch) * np.array(self.Nivel_estoque) -
            custo_fixo_pedido -
            np.array(self.Pedido) * np.array(self.Co)
        )

        return np.sum(reward)
    

    def step(self, action):
        print(action)
        self.Pedido = action.astype(int)
        self.Tempo_entrega = self._get_tempo_entrega()
        self._OnOrder_update(self.Tempo_entrega, self.Pedido, self.lista_em_entrega)
        
        # Remover o primeiro elemento e calcular a soma
        self.Transicao = np.array([lista[0] for lista in self.lista_em_entrega])
        self.lista_em_entrega = [lista[1:] for lista in self.lista_em_entrega]

            
        for i in range(self.num_produtos):
            if np.isnan(self.lista_em_entrega[i]).any():
                # Lide com NaN conforme necessário, por exemplo, substitua por zero
                self.Em_entrega[i] = 0
            else:
                self.Em_entrega[i] = np.nansum(self.lista_em_entrega[i])
            #self.Em_entrega[i] = np.sum(self.lista_em_entrega[i])
        self.Demanda = self._get_demanda()
        demanda_real = self.Demanda
        self.Demanda = np.minimum(self.Demanda, self.Nivel_estoque)
        
        
        self.periodo += 1
        self.Distancia_feriado = self._get_diasferiado()
        self.Dia_semana = self._get_dia_da_semana()
        self.historico = self._get_historico()

        self.produtos_a_mais = np.maximum(self.Nivel_estoque + self.Transicao - self.max_estoque, 0)
        self.backlog = np.maximum(demanda_real - np.minimum(self.Nivel_estoque + self.Transicao, self.max_estoque), 0)
        self.Nivel_estoque = np.maximum(self.Nivel_estoque + self.Transicao - self.Demanda, 0)

        reward = self._calcular_custo()
        done = self.periodo == self.horizonte
        self.state = self._get_observation()

        return self.state, reward, done, {}