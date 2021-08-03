import numpy as np
from numpy import random


num_states = 6

R = np.array([
    [-1, -1, -1, -1, 0, -1],
    [-1, -1, -1, 0, -1, 90],
    [-1, -1, -1, 0, -1, -1],
    [-1, 0, 0, -1, 0, -1],
    [0, -1, -1, 0, -1, 100],
    [-1, 0, -1, -1, 0, 100]
])

Q = np.zeros([num_states, num_states])

gamma = 0.8

num_iterations = 10000
goal_state = 5
np.random.seed = 42

for i in range(num_iterations):
    initial_state = random.randint(num_states-1)
    current_state = initial_state
    possible_actions = [i for i,val in enumerate(R[current_state]) if val>-1]
    action = random.choice(possible_actions)
    qmax = np.amax([val for i,val in enumerate(Q[action]) if R[action,i] > -1])
    Q[current_state, action] = R[current_state, action] + gamma * qmax

    while current_state != goal_state:
        possible_actions = [i for i,val in enumerate(R[current_state]) if val>-1]
        action = random.choice(possible_actions)
        qmax = np.amax([val for i,val in enumerate(Q[action]) if R[action,i] > -1])
        Q[current_state, action] = R[current_state, action] + gamma * qmax
        current_state = action


def determine_path(current_state):
    path = [current_state]
    while current_state != goal_state:
        action = np.argmax(Q[current_state])
        path.append(action)
        current_state = action

    return path


present_room = int(input("Which room are you in? "))
optimal_path = determine_path(present_room)
print(optimal_path)
