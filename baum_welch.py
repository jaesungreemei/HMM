import numpy as np


def alpha_pass(OBS, A, B, Pi):
    '''
    OBS = (1 x k)
    A = (n x n)
    B = (n x m)
    Pi = (1 x n)

    alpha = (n x k)
    '''

    alpha = np.zeros((A.shape[0], OBS.shape[1]))
    alpha[:, [0]] = Pi.transpose() * B[:, OBS[:, 0]]        # Use Pi to calculate for the first column (in accordance to equation)

    for testInd in range(1, OBS.shape[1]):
        for stateInd in range(A.shape[0]):
            alpha[stateInd, testInd] = (alpha[:, [testInd-1]].T).dot(A[:, [stateInd]]) * B[stateInd, OBS[:, testInd]]       # Apply Dynamic Programming Equation

    return alpha


def beta_pass(OBS, A, B):
    '''
    OBS = (1 x k)
    A = (n x n)
    B = (n x m)

    beta = (n x k)
    '''

    beta = np.zeros((A.shape[0], OBS.shape[1]))
    beta[:, OBS.shape[1]-1] = np.ones((A.shape[0]))     # Set the values for the last state to 1

    for testInd in range(OBS.shape[1]-2, -1, -1):
        for stateInd in range(A.shape[0]):
            beta[stateInd, testInd] = (beta[:, testInd+1] * B[:, OBS[:, testInd+1]].T).dot(A[stateInd, :].T)
    
    return beta


def baum_welch(OBS, A, B, Pi, iterNum=100):
    '''
    Expectation-Maximization (EM) Algorithm to optimize the parameters of the HMM such that the model is maximally like the observed data

    OBS = (1 x k)
    A = (n x n)
    B = (n x m)
    Pi = (1 x n)

    alpha = (n x k)
    beta = (n x k)
    '''

    stateNum = A.shape[0]       # n
    testNum = OBS.shape[1]      # k

    for i in range(iterNum):
        alpha = alpha_pass(OBS, A, B, Pi)
        beta = beta_pass(OBS, A, B)

        xi = np.zeros((stateNum, stateNum, testNum-1))
        for t in range(testNum-1):
            print('alpha: {}'.format(str(alpha[[t], :].T.shape)))
            print('dot: {}'.format(str(np.dot(alpha[[t], :].T, A).shape)))
            print('B: {}'.format(str(B[:, OBS[t+1]].T)))

            normalize = np.dot(
                np.dot(alpha[[t], :].T, A) * B[:, OBS[t+1]].T,
                beta[t+1, :]
            )
            for s in range(stateNum):
                update = alpha[t,s] * A[s,:] * B[:, OBS[t+1]].T * beta[t+1, :].T
                xi[i, :, t] = update / normalize
        
        gamma = np.sum(xi, axis=1)
        A = np.sum(xi, 2) / np.sum(gamma, axis=1).reshape((-1, 1))

        gamma = np.hstack((gamma, np.sum(xi[:, :, testNum-2], axis=0).reshape((-1, 1))))

        obsNum = B.shape[1]
        denom = np.sum(gamma, axis=1)
        for j in range(obsNum):
            B[:, j] = np.sum(gamma[:, OBS==1], axis=1)
        
        B = np.divide(B, denom.reshape((-1, 1)))
    
    return {
        "A": A,
        "B": B
    }