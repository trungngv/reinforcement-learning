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
    
random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)

