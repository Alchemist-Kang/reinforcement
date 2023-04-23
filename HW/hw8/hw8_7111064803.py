#
# TD(0) for estimating the state-value function given a policy pi(a|s)
#   Uses parametric function approximation for v -> v(s,w), incremental semi-gradient approach for learning weights
#

#
#"Skeleton" file for HW...
#
# Your task: Complete the missing code (see comments below in the code):
#            1. Implement semi-gradient TD(0) prediction using parametric function approximation for v(s)
#            2. Test your algorithm using the provided td0_v_fa_demo.py file
#

#################################################
#                                               #
#                                               #
#             HW8: Td(0) Semi-gradient          #
#                                               #
#                                               #
#################################################
#
#
## ******************************************** ##
##                                              ##
##  Here use cat_and_mouse.py , td0_v.py for    ##
##  golden answer , hw8_7111064803.py for       ##
##  Semi-gradient , and td0_v_fa_demo.py for    ##
##              running the code.               ##
##                                              ##
## ******************************************** ##

###
### Using python td0_v_fa_demo.py to run the code
###


from random import Random
import numpy as np
import torch

def td0_v_fa(simenv, vmodel, policy, gamma, alpha, num_episodes, max_episode_len, prng_seed=789):
    '''
    Parameters:
        simenv:  Simulation environment instance (e.g., Cat_and_Mouse)
        vmodel:  Model instance for v(s,w) parametric function approximation.
                   Note: Expect a class that has a weights attribute containing
                   a list of all trainable parameter groups (like PyTorch nn classes)
                   and a v(s) method that computes v(s,w) given a state s
        policy:  Policy action-probability matrix, numStates x numActions
        gamma :  Future discount factor, between 0 and 1
        alpha :  Learning step size (used in gradient-descent step)
        num_episodes: Number of episodes to run
        max_episode_len: Maximum allowed length of a single episode
        prng_seed:  Seed for the random number generator       
     '''
    #initialize a few things
    #
    prng=Random()
    prng.seed(prng_seed)
                
    actions=list(range(simenv.numActions)) #assume number of actions is same for all states
    
    #Start episode loop
    #
    for episode in range(num_episodes):
        #use simenv-provided init state
        simenv.initState()
        state=simenv.currentState()
        
        #Run episode sequence to end
        #
        episode_length=0
        while episode_length < max_episode_len:
            #sample the policy actions at current state
            action=prng.choices(actions, weights=policy[state])[0]
            
            #take action, get reward, next state, termination status
            (next_state,reward,term_status)=simenv.step(action)
            
            vmodel_s = vmodel.v(state)  ## compute v(s,w) by sending state
            vmodel_s.backward()         ## Gradient
            with torch.no_grad():
                v_s_next = vmodel.v(next_state)

                for w in vmodel.parameters():
                    if term_status:             ### terminate state v_s_next = 0
                        w+=alpha*(reward-vmodel_s)*w.grad
                    else:
                        w+=alpha*(reward+gamma*v_s_next-vmodel_s)*w.grad
                    w.grad.zero_() #zero gradient no accumulate
            '''        
            #
            #Fill in the missing code here!  Semi-gradient TD(0) using function approximation
            #                                (you should use PyTorch capabilities in your implementation)
            # (Note: test your results with the td0_v_fa_demo.py file)
            #
            '''
            
            #check termination status from environment (reached terminal state?)
            if term_status: break  #if termination status is True, we've reached end of episode
            
            state=next_state
            episode_length+=1
        
    return


