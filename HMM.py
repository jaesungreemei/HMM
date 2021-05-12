import numpy as np
import pandas as pd
from viterbi import viterbi
from baum_welch import alpha_pass, beta_pass, baum_welch

import matplotlib.pyplot as plt

##############################################################################################
# Implement Viterbi Algorithm
'''
EXAMPLE: Let's say we went on a 14-Day Coding Marathon and never went or looked outside. We only know the temperature of our room, whether it was hot or cold. 
Given a 14-Day Sequence of Observations about the weather (i.e. Hot / Cold) and given a HMM = (A, B, Pi), we want to decode the sequence and find what the (hidden) states were during those 14 days.
'''
##############################################################################################

obs_map = {
    'Cold': 0,
    'Hot': 1
}

# 14-Day Sequence of Observations
obs_encoded = [1,1,0,1,0,0,1,0,1,1,0,0,0,1]
OBS = np.array([obs_encoded])

inv_map = dict((a, b) for b, a in obs_map.items())
obs_seq = [inv_map[v] for v in obs_encoded]

obs_states = ['Cold', 'Hot']
hidden_states = ['Snow', 'Rain', 'Sunshine']

# Pi: Initial Probabilities Vector
Pi_encoded = [0, 0.2, 0.8]
Pi = np.array([Pi_encoded])

# A_matrix: Transition Probability Matrix for Hidden States
A_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
A_df.loc[hidden_states[0]] = [0.3, 0.3, 0.4]
A_df.loc[hidden_states[1]] = [0.1, 0.45, 0.45]
A_df.loc[hidden_states[2]] = [0.2, 0.3, 0.5]
print("\n HMM Matrix:\n", A_df)

# B_matrix: Transition Probability Matrix for Observable States
B_df = pd.DataFrame(columns=obs_states, index=hidden_states)
B_df.loc[hidden_states[0]] = [1, 0]
B_df.loc[hidden_states[1]] = [0.8, 0.2]
B_df.loc[hidden_states[2]] = [0.3, 0.7]
print("\n Observable States Matrix:\n", B_df)

A = A_df.values
B = B_df.values

# Implement Algorithm
path, delta, phi = viterbi(Pi, A, B, OBS)
state_map = {0:'Snow', 1:'Rain', 2:'Sunshine'}
state_path = [state_map[v] for v in path]
print(pd.DataFrame().assign(Observation=obs_seq).assign(Best_Path=state_path))


##############################################################################################
# Visualize Probabilities as calculated by Viterbi Algorithm
##############################################################################################

df_delta = pd.DataFrame(delta)
df_phi = pd.DataFrame(phi)

# Maintain stochastic row property by normalization
for col in df_delta.columns:
    df_delta[col] = df_delta[col] / df_delta[col].sum()

df_delta.index = hidden_states
df_delta.columns = [int(x)+1 for x in list(df_delta.columns)]

# Visualize using matplotlib.pyplot
df_delta.T.plot(
    kind = 'bar',
    stacked = True
)


##############################################################################################
# Implement Baum-Welch Algorithm
'''
EXAMPLE: Let's say we didn't know the parameters for the Hidden Markov Model A and B.
Thankfully, we do know the number of hidden states and the number of observable states (i.e. the dimensions of the matrices)
Given a certain sequence of observations, we can thus tune the parameters of the HMM
'''
##############################################################################################

obs_states = [1, 2, 3]      # M = 3
hidden_states = ['H', 'C']      # N = 2

A = np.array([[0.8, 0.1], [0.1, 0.8]])      # Transition Probabilities = (N x N) = (2 x 2)
B = np.array([[0.1, 0.2, 0.7], [0.7, 0.2, 0.1]])        # Emission Probabilities = (N x M) = (2 x 3)
Pi = np.array([[0.5, 0.5]])     # Initial Probabilities = (1 x N) = (1 x 2)

# Test Sequence
OBS = '331122313'
OBS = np.array([[int(x)-1 for x in list(OBS)]])     # Number of Observations = k = 9

# Obtain Parameter Sizes
nStates = len(hidden_states)        # N = 2
tLen = len(OBS)     # k = 9

# Alpha-Pass Algorithm
alpha = pd.DataFrame(alpha_pass(OBS, A, B, Pi))
print("Alpha-Pass Results:")
print(alpha)

# Beta-Pass Algorithm
beta = pd.DataFrame(beta_pass(OBS, A, B))
print("Beta-Pass Results:")
print(beta)

# Baum-Welch Algorithm Implementation
baum_welch_test = baum_welch(OBS, A, B, Pi)

A_result = baum_welch_test['A']
print("Tuned A_matrix Parameters:")
print(A_result)

B_result = baum_welch_test['B']
print("Tuned B_matrix Parameters:")
print(B_result)

# Decode using Viterbi Algorithm using the derived HMM
path, delta, phi = viterbi(Pi, A_result, B_result, OBS)
state_map = {0:'H', 1:'C'}
state_path = [state_map[v] for v in path]
print(pd.DataFrame().assign(Observation=OBS[0]).assign(Best_Path=state_path))


