import numpy as np


def full_reuse(apNum, power):

    action = np.zeros((apNum,2))
    action[:,1] = power

    return action


def TDM_single(apNum, num, apUe, apID, power):

    pos = num % len(apID)
    action = np.zeros((apNum,2))
    ap = apID[pos]
    local_id = np.where(apUe[ap] == pos)[0][0]

    action[ap,0] = local_id
    action[ap,1] = power

    return action

def TDM_multiple(num, apUe, power):

    action = np.zeros((len(apUe),2))
    for i in range(len(apUe)):
        l = len(apUe[i])
        pos = num % l
        action[i,0] = pos
        action[i,1] = power


    return action

