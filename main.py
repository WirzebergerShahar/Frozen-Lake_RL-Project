# -*- coding: utf-8 -*-

from World import World
import numpy as np
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import copy

def Rewards(world, r=-0.04,p=0.8):
     reward=[-1,r,r,r,r,r,-1,r,r,r,r,r,1,-1,-1,r]
     nholes = world.get_stateHoles()
     ngoals = world.get_stateGoal()
     nstates = world.get_nstates()
     nstop=ngoals+nholes
     actions = ["N", "S", "E", "W"]
     nb = [1,5, 9,13]
     wb = [1,2, 3, 4]
     eb = [13,14,15,16]
     sb = [4, 8, 12, 16]
     rewardsa={}
     for a in actions:
         rewards = np.zeros(nstates)
         for i in range(1, nstates + 1):
              if(i in nstop):
                  rewards[i-1]=0
              else:
                if(a=="N"):
                    if(i not in nb):
                        rewards[i - 1]=p*reward[i-2]
                    else:
                        rewards[i - 1] = p * reward[i-1]
                    if(i not in eb):
                        rewards[i - 1] += ((1-p)/2) * reward[i + 3]
                    else:
                        rewards[i - 1] += ((1-p)/2) * reward[i - 1]
                    if (i not in wb):
                        rewards[i - 1] += ((1-p)/2) * reward[i - 5]
                    else:
                        rewards[i - 1] += ((1-p)/2) * reward[i - 1]
                if (a == "S"):
                    if (i not in sb):
                        rewards[i - 1] = p * reward[i]
                    else:
                        rewards[i - 1] = p * reward[i - 1]
                    if (i not in eb):
                        rewards[i - 1] += ((1-p)/2) * reward[i + 3]
                    else:
                        rewards[i - 1] += ((1-p)/2) * reward[i - 1]
                    if (i not in wb):
                        rewards[i - 1] += ((1-p)/2) * reward[i - 5]
                    else:
                        rewards[i - 1] += ((1-p)/2) * reward[i - 1]
                if (a == "E"):
                    if (i not in eb):
                        rewards[i - 1] = p * reward[i+3]
                    else:
                        rewards[i - 1] = p * reward[i - 1]
                    if (i not in nb):
                        rewards[i - 1] += ((1-p)/2) * reward[i-2]
                    else:
                        rewards[i - 1] += ((1-p)/2) * reward[i - 1]
                    if (i not in sb):
                        rewards[i - 1] += ((1-p)/2) * reward[i]
                    else:
                        rewards[i - 1] += ((1-p)/2) * reward[i - 1]
                if (a == "W"):
                    if (i not in wb):
                        rewards[i - 1] = p * reward[i-5]
                    else:
                        rewards[i - 1] = p * reward[i - 1]
                    if (i not in nb):
                        rewards[i - 1] += ((1-p)/2) * reward[i-2]
                    else:
                        rewards[i - 1] += ((1-p)/2) * reward[i - 1]
                    if (i not in sb):
                        rewards[i - 1] += ((1-p)/2) * reward[i]
                    else:
                        rewards[i - 1] += ((1-p)/2) * reward[i - 1]
         rewardsa[a] = pd.DataFrame(rewards)
     return rewardsa


def max_a(transition_models, rewards, gamma, i, V, actions, nstop):
    max_per = {key: 0 for key in actions}
    max_a = ""
    actions = {"N":1, "E":2, "S":3, "W":4}
    for action in actions:
        max_per[action] = 0
        if i not in nstop:
            max_per[action] +=rewards[action].loc[i - 1,0]+ gamma *np.dot(transition_models[action].loc[i, :].values, V)
    maxv = -10e10
    for k in max_per:
        if max_per[k] > maxv:
            max_a = k
            maxv = max_per[k]
    return maxv, actions[max_a]



def value_iter(world, transition_models, rewards, gamma=1.0, theta = 10e-4):
    nstates = world.get_nstates()
    nholes = world.get_stateHoles()
    ngoals = world.get_stateGoal()
    nstop = nholes + ngoals
    value = np.zeros(nstates)
    policy = np.zeros(nstates)
    actions = ["N", "S", "E", "W"]
    delta = 1
    while delta > theta:
        delta = 0
        v = copy.deepcopy(value)
        for i in range(1, nstates + 1):
            value[i - 1], policy[i - 1] = max_a(transition_models, rewards, gamma, i, v, actions, nstop)
            delta = max(delta, np.abs(v[i - 1] - value[i - 1]))
    return value, policy



def policy_iter(world,transition_models,rewards,gamma, theta = 10**-4):
    nactions=world.get_nactions()
    nstates = world.get_nstates()
    policy = np.zeros((nstates, nactions))+1/nactions
    bool = False
    while not bool:
        V = policy_evaluation(world,transition_models,policy, gamma, theta)
        policy_opt = policy_improvement(world,transition_models,rewards,V, gamma)
        world.plot_value(V)
        world.plot_policy(np.argmax(policy_opt, axis=1) + 1)
        for s in range(1, nstates + 1):
            for i in range(1,nactions):
                bool=True
                if (policy_opt[s-1,i-1] != policy[s-1,i-1]):
                    bool = False
        policy = policy_opt
    policy = np.argmax(policy, axis=1) + 1
    return V,policy

def policy_evaluation(world,transition_models,policy, gamma, theta):
    r=-0.04
    reward = [-1, r, r, r, r, r, -1, r, r, r, r, r, 1, -1, -1, r]
    nstates = world.get_nstates()
    nholes = world.get_stateHoles()
    ngoals = world.get_stateGoal()
    nstop = nholes + ngoals
    value = np.zeros(nstates)
    actions = ["N", "E", "S", "W"]
    delta=theta+1
    while delta > theta:
        delta = 0
        v = copy.deepcopy(value)
        for s in range(1, nstates+1):
            mid_sum = 0
            reward_sum=0
            if s not in nstop:
                for action in actions:
                    if action == 'N':
                        a = 1
                    if action == 'E':
                        a = 2
                    if action == 'S':
                        a = 3
                    if action == 'W':
                        a = 4
                    mid_sum += policy[s - 1, a-1] * np.dot(transition_models[action].loc[s, :], value)
                    reward_sum+=policy[s - 1, a-1] * np.dot(reward,transition_models[action].loc[s, :])
            value[s-1] = reward_sum + gamma * mid_sum
            delta = max(delta, np.abs(value[s-1] - v[s-1]))
    return value


def policy_improvement(world, transition_models, rewards, V, gamma):
    nactions=world.get_nactions()
    nstates = world.get_nstates()
    nholes = world.get_stateHoles()
    ngoals = world.get_stateGoal()
    nstop = nholes + ngoals
    actions = ["N", "E", "S", "W"]
    Vs_best=np.zeros(nactions)
    Vs = np.zeros((nstates, nactions))
    policy_opt = np.zeros((nstates, nactions))
    for s in range(1, nstates+1):
        for action in actions:
            if action=='N':
                a=1
            if action == 'E':
                a = 2
            if action=='S':
                a=3
            if action=='W':
                a=4
            Vs[s - 1, a - 1] = rewards[action].loc[s-1,0]
            if s not in nstop:
                Vs[s - 1, a - 1] += np.dot(transition_models[action].loc[s, :].values, V)*gamma
    for s in range(1,nstates+1):
        sumVs=0
        for a in range(1,nactions+1):
            if Vs[s-1, a-1] == np.max(Vs[s-1, :]):
                Vs_best[a-1] = 1
            else:
                Vs_best[a-1] = 0
            sumVs += Vs_best[a-1]
        policy_opt[s-1, :] = Vs_best/sumVs
    return policy_opt





def construct(world, p=0.8):
     actions =["N", "E", "S", "W"]
     nstates = world.get_nstates()
     nrows = world.get_nrows()
     nholes=world.get_stateHoles()
     ngoals=world.get_stateGoal()
     nstop=nholes+ngoals
     nb=[5,9]
     wb=[2,3,4]
     eb=[16]
     sb=[4,8,12,16]
     transition_models={}
     for action in actions:
          transition_model = np.zeros((nstates,nstates))
          for i in range(1,nstates+1):
               if i not in nstop:
                    if action=="N":
                         if i not in nb:
                              transition_model[i-1][i-2] += p
                         else:
                              transition_model[i-1][i-1] += p
                         if i not in wb:
                              transition_model[i-1][i-nrows-1] += (1-p)/2
                         else:
                              transition_model[i-1][i-1] += (1-p)/2
                         if i not in eb:
                              transition_model[i-1][i-1 + nrows] += (1-p)/2
                         else:
                              transition_model[i-1][i-1] += (1-p)/2
                    if action=="S":
                         if i not in sb:
                              transition_model[i-1][i] += p
                         else:
                              transition_model[i-1][i-1] += p
                         if i not in wb:
                              transition_model[i-1][i-nrows-1] +=(1 - p) / 2
                         else:
                              transition_model[i-1][i-1]+=(1 - p) / 2
                         if i not in eb:
                              transition_model[i-1][i+nrows-1]+= (1 - p) / 2
                         else:
                              transition_model[i-1][i-1] += (1 - p) / 2
                    if action=="E":
                         if i not in eb:
                              transition_model[i-1][i +nrows-1] += p
                         else:
                              transition_model[i-1][i-1] += p
                         if i not in nb:
                              transition_model[i-1][i-2] += (1-p)/2
                         else:
                              transition_model[i-1][i-1] += (1-p)/2
                         if i not in sb:
                              transition_model[i-1][i] += (1-p)/2
                         else:
                              transition_model[i-1][i-1] += (1-p)/2
                    if action=="W":
                         if i not in wb:
                              transition_model[i-1][i - nrows-1] += p
                         else:
                              transition_model[i -1][i-1] += p
                         if i not in nb:
                              transition_model[i -1][i-2] += (1 - p) / 2
                         else:
                              transition_model[i-1][i-1] += (1 - p) / 2
                         if i not in sb:
                              transition_model[i-1][i] += (1 - p) / 2
                         else:
                              transition_model[i-1][i-1] += (1 - p) / 2
               else:
                   transition_model[i - 1][i - 1]=1
          transition_models[action] = pd.DataFrame(transition_model, index=range(1, nstates + 1),
                                              columns=range(1, nstates + 1))
     return transition_models




if __name__ == "__main__":

     world = World()
     #world.plot_value([np.random.random() for i in range(world.nStates)])
     #world.plot_policy(np.random.randint(1, world.nActions,(world.nStates, 1)))
     #a
     transition_models = construct(world)
     rewards=Rewards(world)
     N = np.array((rewards['N']))
     S = np.array((rewards['S']))
     E = np.array((rewards['E']))
     W = np.array((rewards['W']))
     mat=np.column_stack((N,S,W,E))
     print('    N      S      W     E')
     print(mat)

     for i, j in transition_models.items():
        print(i)
        print(j)

     #b
     value, policy= value_iter(world, transition_models, rewards)
     world.plot_value(value)
     world.plot_policy(policy)

     #c
     value, policy = value_iter(world, transition_models, rewards, 0.9)
     world.plot_value(value)
     world.plot_policy(policy)

     #d
     rewards1=Rewards(world,-0.02)
     value, policy = value_iter(world, transition_models, rewards1)
     world.plot_value(value)
     world.plot_policy(policy)

     #e
     value, policy= policy_iter(world, transition_models, rewards,0.9)

