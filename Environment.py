import gym
from gym import spaces
from gym.utils import seeding
import random
import numpy as np

class InvOptEnv_unico_produto(gym.Env):
    def __init__(self, max_estoque, estoque_inicial, custo_pedido,custo_manter, penalidade, custo_fixo, time_serie, horizonte):

      self.estoque_inicial = estoque_inicial

      self.Co = custo_pedido   # Custo Pedido
      self.Ch = custo_manter    # Custo de manter estoque
      self.Cp = penalidade        # penalidade por ter demanda e não ter estoque
      self.custo_fixo = custo_fixo

      self.action_space = spaces.Box(low=0, high=max_estoque, shape=(1,), dtype=int)   # -> ações discretas
      self.observation_space = spaces.Tuple((
          spaces.Discrete(max_estoque+1),   # -> Nível do estoque
          spaces.Discrete(max_estoque+1),   # -> Transição de estoque
          spaces.Discrete(max_estoque+1),   # -> Pedido
          spaces.Discrete(max_estoque+1),    # -> Quantidade para chegar
          spaces.Discrete(60),               # -> Distância do feriado;
          spaces.Discrete(7),               # -> Dia_semana;
          spaces.Box(low=0, high=max_estoque, shape=(7,), dtype=np.int32), # -> Histórico de demanda
          ))

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
        #reset the state vector : state = [Nivel_estoque, Transicao, Pedido, Pedido_entrega, Distancia_feriado, historico, dia_semana]
        self.Nivel_estoque = self.estoque_inicial
        self.Transicao = 0
        self.Demanda = 0
        self.Pedido = 0
        self.Em_entrega = 0
        self.Distancia_feriado = 0
        self.Dia_semana = 0
        self.historico = [0]*7
        self.state = self._get_observation()

        #reset tempo entrega
        self.Tempo_entrega = 0

        #reset variables used
        self.backlog = 0
        self.lista_em_entrega = []
        self.produtos_a_mais = 0

        #reset period
        self.periodo = 8


        return self._get_observation()

    #Calcula o tempo de entrega
    def _get_tempo_entrega(self):
        return random.randint(5, 10)

    #obtem a demanda do dia atual
    def _get_demanda(self):
        return self.demand_ts.loc[self.periodo, 'unit_sales']

    #obtem a demanda do dia atual
    def _get_diasferiado(self):
        return self.demand_ts.loc[self.periodo, 'dias_ate_proximo_feriado']

    #obtem a demanda do dia atual
    def _get_dia_da_semana(self):
        return self.demand_ts.loc[self.periodo, 'dia_da_semana']

    #obtem a demanda do dia atual
    def _get_historico(self):
        historico = []
        for i in range(7, 0, -1):
          p = self.periodo - i
          historico.append(self.demand_ts.loc[p, 'unit_sales'])
        return historico

    def _OnOrder_update(self, tempo_entrega, action):
        if len(self.lista_em_entrega) > tempo_entrega:
            self.lista_em_entrega[tempo_entrega] += action
        else:
            self.lista_em_entrega.extend([0]*int(tempo_entrega - len(self.lista_em_entrega) + 1))
            self.lista_em_entrega[tempo_entrega] += action

    #obtem estado atual
    def _get_observation(self):
        obs = (*(self.Nivel_estoque, self.Transicao, self.Pedido, self.Em_entrega, self.Distancia_feriado, self.Dia_semana), *self.historico)
        return np.array(obs).astype(np.int32)

    def _calcular_custo(self):
        if self.Pedido != 0:
            custo_fixo_pedido = 1000
        else:
            custo_fixo_pedido = 0
        return self.custo_fixo + self.Co*self.Pedido + custo_fixo_pedido + self.Ch*self.Nivel_estoque + self.Cp*self.backlog + self.Cp*self.produtos_a_mais

    def step(self, action):
        self.Pedido = action
        self.Tempo_entrega = self._get_tempo_entrega()
        self._OnOrder_update(self.Tempo_entrega, self.Pedido)
        self.Transicao = self.lista_em_entrega.pop(0)
        self.Em_entrega = sum(self.lista_em_entrega)
        self.Demanda = self._get_demanda()
        self.periodo += 1
        self.Distancia_feriado =  self._get_diasferiado()
        self.Dia_semana = self._get_dia_da_semana()
        self.historico =  self._get_historico()
        self.produtos_a_mais = max((self.Nivel_estoque + self.Transicao) - self.max_estoque,0)
        self.backlog = max(self.Demanda - min(self.Nivel_estoque + self.Transicao, self.max_estoque), 0)
        self.Nivel_estoque = max(min(self.Nivel_estoque + self.Transicao, self.max_estoque) - self.Demanda, 0)
        reward = -1*self._calcular_custo()
        done = self.periodo == self.horizonte
        self.state = self._get_observation()

        return self.state, reward, done, {}