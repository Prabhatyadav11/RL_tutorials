# Source: https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa
import gym
import numpy as np
from gym import wrappers

import pdb

# added this to make the frozen lake environment non-slippery i.e. deterministic
# from gym.envs.registration import register
# register(
#     id='FrozenLakeNotSlippery-v0',
#     entry_point='gym.envs.toy_text:FrozenLakeEnv',
#     kwargs={'map_name' : '8x8', 'is_slippery': False},
#     max_episode_steps=100,
#     reward_threshold=0.78, # optimum = .8196
# )


def one_step_lookahead(state, V):

	A = np.zeros(env.nA)
	for a in range(env.nA):
		for p, next_state, reward, done in env.P[state][a]:
			A[a] += p * (reward + gamma * V[next_state])
	return A


def value_iteration(env, gamma = 0.1):
	""" the value-iteration algorithm (sutton's book)"""
	v = np.zeros(env.observation_space.n)
	max_iterations = 100000

	theta = 1e-20

	for i in range(max_iterations):
		prev_v = np.copy(v)
		for s in range(env.nS):
			A = one_step_lookahead(s, v)
			best_action_value = np.max(A)
			v[s] = best_action_value

		if (np.sum(np.fabs(prev_v - v)) <= theta):
			print("Value iteration converged at iteration: %d" %i)
			break
	return v

def extract_policy(v, gamma = 1.0):
	""" Exstract policy given a value-function"""
	policy = np.zeros(env.nS)

	for s in range(env.nS):
		A = one_step_lookahead(s, v)
		best_action = np.argmax(A)
		policy[s] = best_action
	return policy	


def run_episode(env, policy, gamma = 1.0, render = False):

	obs = env.reset()
	total_reward = 0
	step_idx = 0
	while True:
		if render:
			env.render()
		obs, reward, done, _ = env.step(int(policy[obs]))

		total_reward += (gamma ** step_idx * reward)

		step_idx += 1

		if done:
			break
	return total_reward



def evaluate_policy(env, policy, gamma = 1.0, n = 100)	:
	
	total_reward = 0.
	for _ in range(n):
		r = run_episode(env, policy, gamma = gamma, render = True)
		total_reward += r

	return total_reward/n

	


env_name = 'FrozenLake8x8-v0'
gamma = 1
env = gym.make(env_name).unwrapped
optimal_v = value_iteration(env, gamma)
policy = extract_policy(optimal_v, gamma)
policy_score = evaluate_policy(env, policy, gamma, n = 10)
print('policy average score:= %d' %policy_score)

