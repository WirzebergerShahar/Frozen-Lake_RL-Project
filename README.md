# Reinforcement Learning First Project
Markov Decision Process (MDP)- is a mathematical framework for modeling decision making. The decision maker does not have full control on the outcomes of his decisions. MDP is good method for solving optimization problems via Reinforcement Learning or Dynamic Programming.
A MDP consist of 4 main parameters: 
1.	S- is the states available in the problem's environment.
2.	A- is the actions available in the problem's environment.
3.	P- is the probability that action a in state s will lead to the next state s'.
4.	R- is the immediate reward received after the agent move from state s to state s'. 

In this project, we get a penguin on a frozen lake, which is described by a 4x4 grid world with holes and a goal state (fish),
both defining terminal states. For transitions to terminal states the penguin gets a reward of +1 for the
goal state and a reward of −1 for the holes, whereas for all other transitions the penguin gets a reward of
r = −0.04. The penguin can take four possible actions = {N, E, S, W}, but the surface is slippery and only
with probability 0.8 the penguin will go into the intended direction and with probability 0.1 to the left and
0.1 to the right of it. It is assumed that the boundaries are reflective, i.e., if the penguin hits the edge of
the lake it remains in the same grid location. Find the optimal policy for the penguin to get to the fish by
solving the corresponding MDP.


