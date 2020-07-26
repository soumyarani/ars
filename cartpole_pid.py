#%%
#!/usr/bin/python
import gym
import numpy as np
import time 
from cartpole_mod import CartPoleEnv

# Importing the libraries
import os
from gym import wrappers
#import pybullet_envs
'''
#manually tuned gains
kp = 100
kd = 2.5
'''
# Setting the Hyper Parameters

class Hp():
    
    def __init__(self):
        self.nb_steps = 100
        self.episode_length = 200
        self.learning_rate = 0.02
        self.nb_directions = 16
        self.nb_best_directions = 16
        assert self.nb_best_directions <= self.nb_directions
        self.noise = 0.3
        self.seed = 1
        #self.env_name = 'HalfCheetahBulletEnv-v0'

# Normalizing the states

class Normalizer():
    
    def __init__(self, nb_inputs):
        self.n = np.zeros(nb_inputs)
        self.mean = np.zeros(nb_inputs)
        self.mean_diff = np.zeros(nb_inputs)
        self.var = np.zeros(nb_inputs)
    
    def observe(self, x):
        self.n += 1.
        last_mean = self.mean.copy()
        self.mean += (x - self.mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-2)
    
    def normalize(self, inputs):
        obs_mean = self.mean
        obs_std = np.sqrt(self.var)
        return (inputs - obs_mean) / obs_std

# Building the AI

class Policy():
    
    def __init__(self, input_size, output_size):
        self.theta = np.zeros((output_size, input_size))
    
    def evaluate(self, input, delta = None, direction = None):
        if direction is None:
            return self.theta.dot(input)
        elif direction == "positive":
            return (self.theta + hp.noise*delta).dot(input)
        else:
            return (self.theta - hp.noise*delta).dot(input)
    
    def sample_deltas(self):
        return [np.random.randn(*self.theta.shape) for _ in range(hp.nb_directions)]
    
    def update(self, rollouts, sigma_r):
        step = np.zeros(self.theta.shape)
        for r_pos, r_neg, d in rollouts:
            step += (r_pos - r_neg) * d
        if sigma_r != 0:
            self.theta += hp.learning_rate / (hp.nb_best_directions * sigma_r) * step
        print("gains", self.theta)

# Exploring the policy on one specific direction and over one episode

def explore(env, normalizer, policy, direction = None, delta = None, test = None):
    state = env.reset()
    done = False
    num_plays = 0.
    sum_rewards = 0
    while not done and num_plays < hp.episode_length:
        normalizer.observe(state)
        state = normalizer.normalize(state)
        action = policy.evaluate(state, delta, direction)
        
        a = action.item() #env.action_space.sample()
        #print("control:",a)
        if test:
            env.render()
            time.sleep(0.1)
            print(sum_rewards)
        state, reward, done, _ = env.step(a)
        reward = max(min(reward, 1), -1)
        sum_rewards += reward
        num_plays += 1
    env.close()    
    return sum_rewards

# Training the AI

def train(env, policy, normalizer, hp):
    
    for step in range(hp.nb_steps):
        
        # Initializing the perturbations deltas and the positive/negative rewards
        deltas = policy.sample_deltas()
        positive_rewards = [0] * hp.nb_directions
        negative_rewards = [0] * hp.nb_directions
        
        # Getting the positive rewards in the positive directions
        for k in range(hp.nb_directions):
            positive_rewards[k] = explore(env, normalizer, policy, direction = "positive", delta = deltas[k])
        
        # Getting the negative rewards in the negative/opposite directions
        for k in range(hp.nb_directions):
            negative_rewards[k] = explore(env, normalizer, policy, direction = "negative", delta = deltas[k])
        
        # Gathering all the positive/negative rewards to compute the standard deviation of these rewards
        all_rewards = np.array(positive_rewards + negative_rewards)
        sigma_r = all_rewards.std()
        
        # Sorting the rollouts by the max(r_pos, r_neg) and selecting the best directions
        scores = {k:max(r_pos, r_neg) for k,(r_pos,r_neg) in enumerate(zip(positive_rewards, negative_rewards))}
        order = sorted(scores.keys(), key = lambda x:scores[x], reverse = True)[:hp.nb_best_directions]
        rollouts = [(positive_rewards[k], negative_rewards[k], deltas[k]) for k in order]
        
        # Updating our policy
        policy.update(rollouts, sigma_r)
        
        # Printing the final reward of the policy after the update
        if step % 10 == 0:
            reward_evaluation = explore(env, normalizer, policy, test = True)
        reward_evaluation = explore(env, normalizer, policy)    
        print('Step:', step, 'Reward:', reward_evaluation)

# Running the main code

def mkdir(base, name):
    path = os.path.join(base, name)
    if not os.path.exists(path):
        os.makedirs(path)
    return path
work_dir = mkdir('exp', 'brs')
monitor_dir = mkdir(work_dir, 'monitor')

hp = Hp()
np.random.seed(hp.seed)
env = CartPoleEnv()
#env = wrappers.Monitor(env, monitor_dir, force = True)
nb_inputs = env.observation_space.shape[0]
nb_outputs = 1 # env.action_space.shape[0]
policy = Policy(nb_inputs, nb_outputs)
normalizer = Normalizer(nb_inputs)
train(env, policy, normalizer, hp)


'''
#PD code
#%%

def main(): 
    env = CartPoleEnv()
    print(env.action_space)
    env.render()
    #action = env.action_space.sample()

    for i_episode in range(5): 
        policy = Policy(nb_inputs, nb_outputs)
        normalizer = Normalizer(nb_inputs)
        train(env, policy, normalizer, hp)
        observation = env.reset()
        for t in range(200): 
            env.render()
            observation, reward, done, info = env.step(action,0)
            # print(i_episode, t, observation)
            theta      = observation[2]
            error      = theta
            diff_error = observation[3] - observation[1]
            control    = kp*error + kd*diff_error
            if control>0: 
                action = 1
            elif control<0: 
                action = 0
            if done: 
                print("Episode finished after {} timesteps".format(t+1))
                break
            time.sleep(0.01)
        if not done: 
            print("Completed")
    env.close()


if __name__=="__main__": 
    main()
'''

