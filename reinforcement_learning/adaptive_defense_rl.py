import numpy as np

states = 5
actions = 2
q_table = np.zeros((states, actions))

alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.2  # exploration rate

# Dummy environment response: next state and reward
def env_response(state, action):
    next_state = (state + action) % states
    reward = 1 if action == 1 else -1
    return next_state, reward

for episode in range(100):
    state = np.random.randint(states)
    done = False
    while not done:
        if np.random.rand() < epsilon:
            action = np.random.randint(actions)  # explore
        else:
            action = np.argmax(q_table[state])  # exploit
        
        next_state, reward = env_response(state, action)
        q_table[state, action] += alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state
        done = episode > 90

print('Q-table:', q_table)
