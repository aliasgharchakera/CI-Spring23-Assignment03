import numpy as np

# Define the grid world
N = 5
grid = np.zeros((N, N))
# Set the obstacles
grid[1, 1] = -100
grid[2, 3] = -100
# Set the rewards
grid[0, 3] = 100
grid[4, 2] = 100

# Define the state space
state_space = [(i, j) for i in range(N) for j in range(N)]

# Define the action space
action_space = ['Up', 'Down', 'Left', 'Right']

# Define the reward function
def reward(state):
    i, j = state
    if grid[i, j] == -100:
        return -100
    elif grid[i, j] == 100:
        return 100
    else:
        return 0

# Define the transition function
def transition(state, action):
    i, j = state
    if action == 'Up':
        if i == 0 or grid[i-1, j] == -100:
            return state
        else:
            return (i-1, j)
    elif action == 'Down':
        if i == N-1 or grid[i+1, j] == -100:
            return state
        else:
            return (i+1, j)
    elif action == 'Left':
        if j == 0 or grid[i, j-1] == -100:
            return state
        else:
            return (i, j-1)
    elif action == 'Right':
        if j == N-1 or grid[i, j+1] == -100:
            return state
        else:
            return (i, j+1)

# Define the value function
V = np.zeros((N, N))

# Define the policy
def policy(state, T):
    i, j = state
    values = []
    for action in action_space:
        next_state = transition(state, action)
        values.append(V[next_state[0], next_state[1]])
    values = np.array(values)
    prob = np.exp(values / T) / np.sum(np.exp(values / T))
    return np.random.choice(action_space, p=prob)

# Train the agent
alpha = 0.1 # learning rate
gamma = 0.9 # discount factor
T = 1.0 # temperature
num_episodes = 1000
for episode in range(num_episodes):
    state = (np.random.randint(N), np.random.randint(N))
    while True:
        action = policy(state, T)
        next_state = transition(state, action)
        r = reward(next_state)
        V[state[0], state[1]] += alpha * (r + gamma * V[next_state[0], next_state[1]] - V[state[0], state[1]])
        state = next_state
        if r != 0 or np.random.rand() < 0.1: # end of episode
            break

# Evaluate the agent
num_episodes = 100
total_reward = 0
for episode in range(num_episodes):
    state = (np.random.randint(N), np.random.randint(N))
    episode_reward = 0
    while True:
        action = policy(state, T)
        next_state = transition(state, action)
        r = reward(next_state)
        episode_reward += r
        state = next_state
        if r != 0 or np.random.rand() < 0.1: # end of episode
            break
    total_reward += episode_reward
average_reward = total_reward / num_episodes
print("Average reward:", average_reward)

# Print the value function
print("Value function:")
for i in range(N):
    for j in range(N):
        print("%.2f" % V[i, j], end=" ")
    print()
