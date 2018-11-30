
# Implementation of Q-learning (tabular) method for Frozen-lake environment
# Ref: https://github.com/simoninithomas/Deep_reinforcement_learning_Course/blob/master/Q%20learning/FrozenLake/Q%20Learning%20with%20FrozenLake.ipynb
import gym
import numpy as np
import random
from gym import wrappers
import matplotlib
import matplotlib.pyplot as plt


# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()


env_name = 'FrozenLake-v0' # delete 8x8 to run 4x4 env
gamma = 1
env = gym.make(env_name).unwrapped

# Create a Q-table
q_table = np.zeros((env.nS, env.nA))

total_episodes = 5000
alpha = 0.8
max_steps = 99
gamma = 0.95

# Exploration parameters
epsilon = 1.0
max_epsilon = 1.0
min_epsilon = 0.01
decay_rate = 0.005




def plot_durations():
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.plot(rewards)
    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())




rewards = []

for episode in range(total_episodes):
	state = env.reset()
	step = 0
	total_rewards = 0

	for steps in range(max_steps):

		tradeoff_factor = random.uniform(0,1)

		if tradeoff_factor > epsilon:
			action = np.argmax(q_table[state]) # Exploitation

		else:
			action = env.action_space.sample() # Exploration

		next_state, reward, done, _ = env.step(action)


		q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * np.max(q_table[next_state, :]))

		total_rewards += reward

		state = next_state

		if done:
			rewards.append(total_rewards)
			plot_durations()
			break

	epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)



# Use the learnt Q-table for inference
for episode in range(5): # evaluate five times
	
	state = env.reset()
	print("****Episode: %d****" %episode)

	for step in range(max_steps):
		action = np.argmax(q_table[state, :])
		next_state, reward, done, _ = env.step(action)

		if done:
			env.render()
			print("number of steps: %d" %step)
			break;

		state = next_state

env.close()








