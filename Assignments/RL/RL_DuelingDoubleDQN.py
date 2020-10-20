import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Add, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers.experimental.preprocessing import Normalization

import os
import game2048_env
import gym
import argparse
import numpy as np
from collections import deque
import random

tf.keras.backend.set_floatx('float64')
wandb.init(name='DuelingDoubleDQN', project="deep-rl-2048")

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--eps', type=float, default=1.0)
parser.add_argument('--eps_decay', type=float, default=0.995)
parser.add_argument('--eps_min', type=float, default=0.01)

args = parser.parse_args()

class ReplayBuffer:
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape((args.batch_size, 4, 4, 16))
        next_states = np.array(next_states).reshape((args.batch_size, 4, 4, 16))
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)

class ActionStateModel:
    def __init__(self, state_dim, aciton_dim):
        self.state_dim = state_dim
        self.action_dim = aciton_dim
        self.epsilon = args.eps
        
        self.model = self.create_model()
    
    def create_model(self):
        # normalization = Normalization()
        # adapt_data = np.array([0., 2., 4., 8., 16., 32., 64., 128., 256., 512., 1024., 2048.], dtype=np.float32)
        # normalization.adapt(adapt_data)

        inputs = Input((self.state_dim))
        # inputs = Input((INPUT_SHAPE_DNN,))
        flatten = Flatten()(inputs)
        # backbone = normalization(inputs)
        backbone = Dense(512, activation='relu')(flatten)
        backbone = Dense(512, activation='relu')(backbone)
        backbone = Dense(4096, activation='relu')(backbone)
        backbone = Dense(4096, activation='relu')(backbone)
        backbone = Dense(4096, activation='relu')(backbone)

        value_output = Dense(1)(backbone)
        advantage_output = Dense(self.action_dim)(backbone)
        output = Add()([value_output, advantage_output])
        model = tf.keras.Model(inputs, output)
        model.compile(loss='mse', optimizer=Adam(args.lr))
        return model
    
    def predict(self, state):
        return self.model.predict(state)
    
    def get_action(self, state):
        # state = np.reshape(state, [1, self.state_dim])
        self.epsilon *= args.eps_decay
        self.epsilon = max(self.epsilon, args.eps_min)
        q_value = self.predict(state)[0]
        if np.random.random() < self.epsilon:
            return random.randint(0, self.action_dim-1)
        return np.argmax(q_value)

    def train(self, states, targets):
        self.model.fit(states, targets, epochs=1, verbose=0, callbacks=[wandb.keras.WandbCallback()])
    

class Agent:
    def __init__(self):
        env = game2048_env.Game2048Env()
        env.seed(0)
        # env.set_max_tile(2048)
        
        self.env = env
        # self.state_dim = 16
        # self.action_dim = 4
        # self.env = gym.make("CartPole-v1")
        self.state_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n

        self.model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_model = ActionStateModel(self.state_dim, self.action_dim)
        self.target_update()

        self.buffer = ReplayBuffer()

    def target_update(self):
        weights = self.model.model.get_weights()
        self.target_model.model.set_weights(weights)
    
    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, done = self.buffer.sample()
            targets = self.target_model.predict(states)
            next_q_values = self.target_model.predict(next_states)[range(args.batch_size),np.argmax(self.model.predict(next_states), axis=1)]
            targets[range(args.batch_size), actions] = rewards + (1-done) * next_q_values * args.gamma
            self.model.train(states, targets)
    
    def train(self, max_episodes=1000):
        for ep in range(max_episodes):
            done, total_reward = False, 0
            state = self.env.reset()
            highest_tile = 0
            while not done:
                action = self.model.get_action(state)
                next_state, reward, done, info = self.env.step(action)
                if info['highest'] > highest_tile:
                    highest_tile = info['highest']
                self.buffer.put(state, action, reward*0.01, next_state, done)
                total_reward += reward
                state = next_state
            
            if self.buffer.size() >= args.batch_size:
                self.replay()                
            self.target_update()
            print('EP{} EpisodeReward={}'.format(ep, total_reward))
            wandb.log({'Reward': total_reward, 'Highest tile': highest_tile})


def main():
    agent = Agent()
    agent.train(max_episodes=1_000)
    agent.model.model.save(os.path.join(wandb.run.dir, "model.h5"))

if __name__ == "__main__":
    main()