import sys
import os
import numpy as np
import gym
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logger import Logger
from memory.PrioritizedReplayBuffer import PrioritizedReplayBuffer
from trainer.Trainer import Trainer
from trainer.Tester import Tester
from models.Q_Network import Q_Network
from utils.setSeed import set_seed
from utils.assertWeightsSync import assert_weight_sync
from utils.init_optimiser import init_optimiser

class DQNAgent:
    def __init__(self, buffer_size, batch_size, episodes, input_shape=(84,84,4), action_size=2,
                gamma=0.9, epsilon=0.9, epsilon_min=0.1, epsilon_decay=0.999, learning_rate=0.0001, 
                tau=0.001, n=3, optimiser='Adam', environment=None, update_frequency=100, wandb_trigger=False,
                load_baseline=False,weights=None ):
        
        self._set_hyperparameters(buffer_size=buffer_size, batch_size=batch_size, episodes=episodes, input_shape=input_shape,
                                  action_size=action_size, gamma=gamma, epsilon=epsilon, epsilon_min=epsilon_min, epsilon_decay=epsilon_decay,
                                  learning_rate=learning_rate, tau=tau, optimiser=optimiser, update_frequency=update_frequency,
                                  wandb_trigger=wandb_trigger, load_baseline=load_baseline, weights=weights,n=n)

        self._init_environment(environment)
        self._init_networks(load_baseline=load_baseline,weights=weights)
        self.replay_buffer = PrioritizedReplayBuffer(state_size=self.input_shape,
                                                     buffer_size=self.buffer_size, batch_size=self.batch_size,n=self.n)
        self.logger = Logger(config={
            "project_name": "DemonAttack_V2",
            "buffer_size": self.buffer_size,
            "batch_size": self.batch_size,
            "episodes": self.episodes,
            "input_shape": self.input_shape,
            "action_size": self.action_size,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "learning_rate": self.learning_rate,
            "tau": self.tau,
            "optimizer": str(self.optimiser),
            "update_frequency": self.update_frequency
        }, env=self.env, model=self.qnet.model , use_wandb=self.wandb_trigger)
        self.trainer = Trainer(self.env,self.episodes,self.replay_buffer,
                               self.logger,self.epsilon, self.epsilon_min, self.epsilon_decay,
                               self.qnet,self.target_net,self.action_size,self.batch_size,
                               self.gamma,self.tau,self.optimiser,self.update_frequency)
        self.tester = Tester(self.qnet,self.env,self.logger,10,self.action_size)
        

    def _set_hyperparameters(self, buffer_size, batch_size, episodes, input_shape, action_size,
                         gamma, epsilon, epsilon_min, epsilon_decay, learning_rate, 
                         tau, optimiser, update_frequency, wandb_trigger,load_baseline,weights,n):
        set_seed()
        self.input_shape = input_shape
        self.action_size = action_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.episodes = episodes
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate
        self.tau = tau
        self.optimiser = init_optimiser(optimiser,learning_rate=learning_rate)
        self.update_frequency = update_frequency
        self.wandb_trigger = wandb_trigger
        self.load_baseline = load_baseline
        self.weights = weights
        self.n=n



    def _init_networks(self,load_baseline=False,weights=None):
        if load_baseline == False:
            self.qnet = Q_Network(self.input_shape, self.action_size,  learning_rate=self.learning_rate ,optimizer=self.optimiser)
            self.target_net = Q_Network(self.input_shape, self.action_size, learning_rate=self.learning_rate ,optimizer=self.optimiser)
            self.target_net.model.set_weights(self.qnet.model.get_weights())
        else:
            self.qnet = Q_Network(self.input_shape, self.action_size, learning_rate=self.learning_rate ,optimizer=self.optimiser)
            self.qnet.model.load_weights(weights)
            self.target_net = Q_Network(self.input_shape, self.action_size, learning_rate=self.learning_rate ,optimizer=self.optimiser)
            self.target_net.model.set_weights(self.qnet.model.get_weights())
        assert_weight_sync(self.qnet.model,self.target_net.model)

    def _init_environment(self, environment):
        if environment:
            self.env = environment
            self.env_name = self.env.env_id()
        else:
            self.env = gym.make("Blackjack-v1", sab=True)
            self.env_name = self.env.env_id()
    
    def initiate_training(self):
        self.trainer.train()
    
    def test_agent(self, weights_path):
        self.tester.test(weights_path)



