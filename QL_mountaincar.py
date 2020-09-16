# STOLEN FROM:
    # https://youtu.be/Gq1Azv_B4-4

import gym
import numpy as np
import pandas as pd
import datetime

# THIS WORKS BUT MOSTLY HARDCODED.
# QUESTIONS:

# how can i view an overall score over time? (learning curve / convergence)
# how can i implement early stopping and how to decide when?

# user params
learning_rate = 0.1 # multplier for weighting rate of change
discount = 0.95 # weighting between current vs future reward
episodes = 20000 # something big

# a quick note on epsilon:
# epsilon is introduced as a probability of taking random actions
# the probability decays (reduces) according to the start and stop episodes
epsilon = 0
epsilon_start_pos = 1
epsilon_end_pos = (episodes + 1) // 2
epsilon_decay_val = epsilon / (epsilon_end_pos / epsilon_start_pos)
showable = 200

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

discrete_state = get_discrete_state(env.reset())
success = False
counter = 0
current_best = 10000
worst_moves = 200

for episode in range(episodes + 1):

    print("Episode N: {} Epsilon value: {:.5f} State: {} Success: {} Best: {:.4f}".format(
        episode, epsilon, discrete_state, success, current_best/worst_moves))

    if episode % showable == 0:
        will_render = True
    else:
        will_render = False
        
    # this is all one episode, taking 200 steps.
    discrete_state = get_discrete_state(env.reset())
    done, success = False, False
    counter = 0
    while not done:
        counter += 1
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, n_actions)
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
            success = True
            q_table[discrete_state + (action, )] = 0
        
        discrete_state = new_discrete_state
    if counter < current_best:
        current_best = counter
    if epsilon_end_pos >= episode >= epsilon_start_pos and epsilon > 0:
        epsilon -= epsilon_decay_val
        
env.close()