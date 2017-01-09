# Value iteration implementation

import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()

def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
    # Reminder: generalised approach is to do policy evalution and policy improvement iteratively until converged
    # With value iteration, we update policy as soon as we update value of a state. This is stochastic / online 
    # instead of batch update.

    
    def one_step_lookahead(state, V):
        """
        Helper function computes one step look ahead for all action in a given state
        
        Returns:
            A vector containing expected value of each action
        """
        A = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, next_state, reward, done in env.P[state][a]:
                A[a] += prob * (reward + discount_factor * V[next_state])
                
        return A

    V = np.zeros(env.nS)
    iter_cnt = 0
    
    while True:
        iter_cnt += 1
        delta = 0
        # for each state
        for s in range(env.nS):
            A = one_step_lookahead(s, V)
            best_action_value = np.max(A)
            # calculate delta across all states seen so far
            delta = max(delta, np.abs(best_action_value - V[s]))
            # Greedily set value of state to value of best action (not the expected value like in policy evaluation)
            # TODO: what if we get expected value instead of the best action value?
            V[s] = best_action_value
            
        # Can we stop?
        if delta < theta:
            break

    # Find optimal policy from value iteration
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        best_action = np.argmax(one_step_lookahead(s, V))
        policy[s, best_action] = 1.0
    
    print("Total iterations: ", iter_cnt)
    return policy, V

policy, v = value_iteration(env)

print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")

expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

