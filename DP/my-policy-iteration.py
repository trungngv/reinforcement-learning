# Policy evaluation implementation
import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv

env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.nS)
    iter = 0
    while True:
        delta = 0
        iter = iter + 1
        # compute value for each state
        for s in range(env.nS):
            v = 0
            # Bellman update (full backup) to compute expected value
            for action, action_prob in enumerate(policy[s]):
                # Know full-dynamic of the environment p(state', reward | action, state),
                # so can use it to compute exact update
                for prob, next_state, reward, done in env.P[s][action]:
                    v += action_prob * prob * (reward + discount_factor * V[next_state])
            # check if value function has converged    
            delta = max(delta, np.abs(V[s] - v))
            V[s] = v
        if delta < theta:    
            break
          
    print('Policy evaluation converges in {} iterations', iter)
    return np.array(V)

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        # Compute optimal value function for the given policy
        V = policy_eval(policy, env, discount_factor)
        
        # Policy is not stable if we make any changes to the policy
        policy_stable = True
        
        # Compute new greedy policy given new value function
        for s in range(env.nS):
            # Current best action
            current_best_a = np.argmax(policy[s])

            # Find the value for each action
            action_values = np.zeros(env.nA)
            # This is only 1-step look-ahead (coarse approximation), deeper look-ahead will lead to better approximation
            # and thus overall performance
            for a in range(env.nA):
                for prob, next_state, reward, done in env.P[s][a]:
                    action_values[a] += prob * (reward + discount_factor * V[next_state])
            best_a = np.argmax(action_values)
            if current_best_a != best_a:
                policy_stable = False
                
            # Quick way to set policy to the best    
            policy[s] = np.eye(env.nA)[best_a]
            
        if policy_stable:
            return policy, V
    
policy, v = policy_improvement(env)
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

# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)
