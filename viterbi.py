import numpy as np

def viterbi(pi, a, b, obs):
    '''
    pi = (1 x nStates)
    a = (nStates x nStates)
    b = (nStates x m)
    obs = (1 x k)

    path = (1 x k)
    delta = (nStates x T)
    phi = (nStates x T)
    '''

    nStates = np.shape(b)[0]        # Number of Hidden States
    k = np.shape(obs)[1]        # Length of Observed Sequence

    # Initialize Blank Path
    path = np.zeros(k, dtype=int)       # Zero Vector of Length k --> (1 x k)
    delta = np.zeros((nStates, k))      # Record highest probability of any path that reaches state k --> (nStates x k)
    phi = np.zeros((nStates, k))        # Record argmax by time step for each state --> (nStates x k)

    # Initialize delta, phi
    delta[:, [0]] = pi.transpose() * b[:, obs[:, 0]]        # (nStates x 1)
    phi[:, 0] = 0

    # Forward Algorithm (alpha-pass)
    print('\nStart Forward Walk\n')
    for t in range(1,k):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[:, t]]     # Instead of summing the values up, pick the highest probability
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
            print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s,t]))
    
    # Backtrack to find optimal path
    print('-'*50)
    print('Finding Optimal Path...\n')
    path[k-1] = np.argmax(delta[:, k-1])
    for t in range(k-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
        print('path[{}] = {}'.format(t, path[t]))
    
    return path, delta, phi