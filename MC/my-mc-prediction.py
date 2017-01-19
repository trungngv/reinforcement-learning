import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict

if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()

def mc_prediction(policy, env, num_episodes, discount_factor=1.0):
    """
    Monte Carlo prediction algorithm (first-visit state). Calculates the value function
    for a given policy using sampling.
    
    Args:
        policy: A function that maps an observation to action probabilities.
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
    
    Returns:
        A dictionary that maps from state -> value.
        The state is a tuple and the value is a float.
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    first_visits_count = defaultdict(float)
    rewards_sum = defaultdict(float)
    
    # The final value function
    # V[s] = reward_sum[s] / first_visit_count[s]
    V = defaultdict(float)
    
    for i in range(1, num_episodes + 1):
        # Generate an episode
        state = env.reset()
        episode = []
        visited_states = set()
        while True:
            # add this state to the visited set of the episode
            visited_states.add(state)
            probs = policy(state)
            action = np.random.choice(len(probs), p = probs)
            next_state, reward, done, _ = env.step(action)
            episode.append((state, action, reward))
            if done:
                break
            state = next_state
            
        # for every unique state that was visited: increase their count and add total reward
        # from episode (G[s]) to the total sums
        for state in visited_states:
            first_occurence_idx = next(i for i,x in enumerate(episode) if x[0] == state)
            first_visits_count[state] += 1.0
            rewards_sum[state] += sum([x[2] for x in episode[first_occurence_idx:]])
            # online update
            # V[state] = returns_sum[state] / returns_count[state]

    for state, reward in rewards_sum.items():
        V[state] = reward / first_visits_count[state]
        
    return V

def sample_policy(observation):
    """
    A policy that sticks if the player score is > 20 and hits otherwise.
    """
    score, dealer_score, usable_ace = observation
    return np.array([1.0, 0.0]) if score >= 20 else np.array([0.0, 1.0])

np.random.seed(1110)    
V_10k = mc_prediction(sample_policy, env, num_episodes=10000)
plotting.plot_value_function(V_10k, title="10,000 Steps")

np.random.seed(1110)
V_50k = mc_prediction(sample_policy, env, num_episodes=50000)
plotting.plot_value_function(V_50k, title="50,000 Steps")
