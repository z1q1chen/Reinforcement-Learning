# !apt-get install -y xvfb python-opengl > /dev/null 2>&1
# !pip install gym pyvirtualdisplay > /dev/null 2>&1

import gym
import numpy as np
import random
import math
from tqdm import tqdm

# We can also use CartPole-v1 but it's slower
env_name = "CartPole-v0"

# Information from OpenAi Gym's repository, velocity and tip_velocity should be 
# -inf to inf but for simplification I just used the values below
position_limit = [-2.4, 2.4]
velocity_limit = [-2.5, 2.5]
angle_limit = [-0.209, 0.209]
tip_velocity_limit = [-1.0, 1.0]

# Paramters representing how many intervals we discretize the state
n_p, n_v, n_a, n_t = z,40,40,40

p = np.linspace(position_limit[0], position_limit[1], n_p)
v = np.linspace(velocity_limit[0], velocity_limit[1], n_v)
a = np.linspace(angle_limit[0], angle_limit[1], n_a)
t = np.linspace(tip_velocity_limit[0], tip_velocity_limit[1], n_t)

def state_to_string(s, p = p, v = v, a = a, t = t):
  position = np.searchsorted(p, s[0], "right")
  velocity = np.searchsorted(v, s[1], "right")
  angle = np.searchsorted(a, s[2], "right")
  tip_velocity = np.searchsorted(t, s[3], "right")
  return f'{position}_{velocity}_{angle}_{tip_velocity}'

def generate_target_policy(Q, states):
  # Target policy based on state action values
  return {s: max(Q[s], key=Q[s].get) for s in states}

def generate_behavior_policy(pi, actions, states, epsilon = 0.5):
  # e-soft policy that makes sure each action is taken at least occasionally
  # This is the behavior policy, it's generated from existing target policy
  b = {s: {a: 1 - epsilon + epsilon/len(actions) if pi[s] == a else 
           epsilon/len(actions) for a in actions} for s in states}
  return b

# Generate episodes
def generate_episode(b, actions):
  # Reset environment
  s = env.reset()
  s = state_to_string(s)
  E = []
  for i in range(200):
    # Choose action based on behavior policy
    action = np.random.RandomState(0).choice(actions, p=list(b[s].values()))
    next_s, next_r, done, info = env.step(action)
    # Add state, action and reward to episodes
    E.append({"state": s, "action": action, "next_reward": next_r})
    if done:
      break
    s = state_to_string(next_s)
  return E

# Test target policy
def test(target_policy):
  env = gym.make(env_name)
  s = env.reset()
  s = state_to_string(s)
  for i in range(201):
    action = target_policy[s]
    obs, reward, done, info = env.step(action)
    s = state_to_string(obs)
    if done:
      break
  env.close()
  return i

env = gym.make(env_name)

# Initialize all states
states = [f'{p}_{v}_{a}_{t}' for p in range(n_p+1) for v in range(n_v+1) for a in 
          range(n_a+1) for t in range(n_t+1)]

# Possible actions
actions = list(range(env.action_space.n))

# Initialize Q and C, for all s in S, a in A(s)
Q = {s:{a: random.uniform(0,1) for a in actions} for s in states}
C = {s:{a: 0 for a in actions} for s in states}

target_policy = generate_target_policy(Q, states)

# Some hyper parameters
n_iterations = 100000
decay = 0.95

for k in tqdm(range(n_iterations), position=0, leave=True):
  # We can also use dynamic epsilon, but in this case I'm using a constant
  epsilon = 0.2
  b = generate_behavior_policy(target_policy, actions, states, epsilon = epsilon)
  E = generate_episode(b, actions)
  G = 0.0
  W = 1.0
  # Reverse time order
  for t in range(len(E) - 1, 0, -1):
    s, a, r = E[t]['state'], E[t]['action'], E[t]['next_reward']
    G = decay * G + r
    C[s][a] = C[s][a] + W
    Q[s][a] = Q[s][a] + (W/C[s][a]) * (G - Q[s][a])
    target_policy[s] = max(Q[s], key=Q[s].get)
    # print(f'{s}-----{target_policy[s]}')
    if a != target_policy[s]:
      break
    W = W /b[s][a]
env.close()

print(f'test ran {test(target_policy)} iterations')