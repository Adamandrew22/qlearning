# STOLEN FROM:
    # https://youtu.be/Gq1Azv_B4-4

import gym
import numpy as np
import pandas as pd
import datetime

# THIS WORKS BUT MOSTLY HARDCODED. QUESTIONS:

# what am i trying to optimise?
# how can i view an overall score over time? (learning curve / convergence)
# how can i implement early stopping and how to decide when?
# need to incorporate an epsilon value to introduce random exploratory actions



# user params
learning_rate = 0.1 # multplier for weighting rate of change
discount = 0.95 # weighting between current vs future reward
episodes = 5000 # something big

epsilon = 0.5
epsilon_start_pos = 1
epsilon_end_pos = 1000
epsilon_decay_val = epsilon / (epsilon_end_pos / epsilon_start_pos)
showable = 500



env = gym.make("MountainCar-v0")
env.reset()

# environment space has 2 dimensions, hence optimal x and y coordinates
# for this example, the state is a list of [position, velocity]
n_actions = env.action_space.n
high_vals = env.observation_space.high
low_vals  = env.observation_space.low
goal_state = [env.goal_position, env.goal_velocity]
discrete_os_size = [20] * len(env.observation_space.high)
discrete_os_size_win = (high_vals - low_vals) / discrete_os_size
q_table = np.random.uniform(low = -2, high = 0, size = (discrete_os_size + [n_actions]))

def get_discrete_state(state):
    discrete_state = (state - low_vals) / discrete_os_size_win
    return tuple(discrete_state.astype(np.int))


for episode in range(episodes):
    print("Episode N: {}".format(episode))
    if episode % showable == 0:
        will_render = True
    else:
        will_render = False
        
    # this is all one episode, taking 200 steps.
    discrete_state = get_discrete_state(env.reset())
    done = False
    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.randm.randint(0, n_actions)
        new_state, reward, done, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        if will_render == True:
            env.render()
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action, )]
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state + (action, )] = new_q
        elif new_state[0] >= goal_state[0]:
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state
    
    if epsilon_end_pos >= episode >= epsilon_start_pos:
        epsilon -= epsilon_decay_val
        
env.close()