# https://gym.openai.com/docs
import tensorflow as tf
import tflearn

import gym

# Some available environments:
# - MountainCar-v0,
# - MsPacman-v0 (requires the Atari dependency),
# - Hopper-v1 (requires the MuJoCo
# List of all environments
from gym import envs
print(envs.registry.all())

# env is the key class with following properties and methods
# - step: take action, receive next state (observation and reward)
# - action_space: space of possible actions
#     - sample(): take a random action
# - observation_space: space of possible states
#     - sample() get a sample state
#      - high(): high state
#      - low(): low state
      
# Spaces introspection
from gym import spaces
space = spaces.Discrete(8)
x = space.sample()
assert space.contains(x)

env = gym.make('MsPacman-v0')
print(env.action_space)
#> Discrete(2)
print(env.observation_space)
#> Box(4,)

# Agent would need to learn the following things
# - Optimal state value function v(s) for all s in observation_space
# -   or optimal state-action value function q(s, a) for all s in observation and a in action
# - Optimal policy is derived from the optimal value function
# - Transition probabilities if using dynamic programming (exact solution)

# Recording episodes
env = gym.make('CartPole-v0')
env.monitor.start('cartpole-experiment-1')
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

env.monitor.close()
