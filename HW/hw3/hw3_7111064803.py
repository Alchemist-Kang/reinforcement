#################################################
#                                               #
#                                               #
#   HW3:   2-D Gridworld Model with iterative   #
#          method and find the optimal policy   #
#                                               #
#                                               #
#################################################

import numpy as np
from enum import IntEnum
import matplotlib.pyplot as plt
import copy

#numpy print options
np.set_printoptions(precision=2)
np.set_printoptions(threshold=np.inf) 

nstates = 25
nactions = 4 
gamma=0.9
    
class Action(IntEnum):
    North=0
    South=1
    West=2
    East=3
A=Action #alias for shorter names!


def veval_matrix(pssa, rsa, policy, gamma):
    #number of states & actions
    nstates=rsa.shape[0]
    nactions=rsa.shape[1]
        
    #First, let's compute a few useful intermediate matrices,
    # r(s,a) and p(s',s,a) averaged over policy actions    
    Rpi=np.sum(policy*rsa,axis=1)
    Rpi.shape=(nstates,1) #reshape into column vector
    
    Ppi=np.zeros((nstates,nstates),dtype=np.float32)
    for j in range(nstates):
        Ppi[j]=np.sum(policy[j]*pssa[j],axis=1)
    
    
    #solve for value function using system of linear eqns: v=(I-Ppi)^-1*Rpi
    #
    ident=np.identity(nstates, dtype=np.float32)
    v=np.linalg.inv(ident-gamma*Ppi)@Rpi
    
    #return computed value function
    return v

def val_iter_syn(v,pssa, rsa, policy, gamma):
    #number of states & actions
    nstates=rsa.shape[0]
    nactions=rsa.shape[1]
    v_current = v
    v_next = np.zeros(shape=(nstates,1))
    delta = 0
    theta = 1e-4
        
    #First, let's compute a few useful intermediate matrices,
    # r(s,a) and p(s',s,a) averaged over policy actions    
    Rpi=np.sum(policy*rsa,axis=1)
    Rpi.shape=(nstates,1) #reshape into column vector

    # iteration0:
    Ppi=np.zeros((nstates,nstates),dtype=np.float32)
    for j in range(nstates):
        Ppi[j]=np.sum(policy[j]*pssa[j],axis=1)

    v_next = Rpi + gamma*Ppi@v_current
    delta = np.max(np.abs(v_current-v_next))            # update delta


    while(delta > theta):
        for j in range(nstates):
            Ppi[j]=np.sum(policy[j]*pssa[j],axis=1)

        v_current = v_next
        v_next = Rpi + gamma*Ppi@v_current

        delta = np.max(np.abs(v_current-v_next))        # update delta
    
    return v_next

def q_pi_sa(rsa, gamma, policy, Vpi):
    nstates=rsa.shape[0]
    nactions=rsa.shape[1]
    q_pi_sa = np.zeros((nstates,nactions),dtype=np.float32)
    for i in range(nstates):
        for j in range(nactions):
            if((i == 1)):         ###(special A to A')
                q_pi_sa[i,j] = rsa[i,j] + gamma*Vpi[21]
            if(i == 3):         ###(special B to B')
                q_pi_sa[i,j] = rsa[i,j] + gamma*Vpi[13]
                
            if((i != 1) and (i != 3)):
                if(j == 0): ###(Mean action is North)
                    if(i<5):
                        q_pi_sa[i,j] = rsa[i,j] + gamma*Vpi[i]
                    else:    
                        q_pi_sa[i,j] = rsa[i,j] + gamma*Vpi[i-5]
                if(j == 1): ###(Mean action is South)
                    if((i<25) and (i>19)):
                        q_pi_sa[i,j] = rsa[i,j] + gamma*Vpi[i]
                    else:    
                        q_pi_sa[i,j] = rsa[i,j] + gamma*Vpi[i+5]
                
                if(j == 2): ###(Mean action is West)
                    if((i==0) or (i==5) or (i==10) or (i==15) or (i==20)):
                        q_pi_sa[i,j] = rsa[i,j] + gamma*Vpi[i]
                    else:    
                        q_pi_sa[i,j] = rsa[i,j] + gamma*Vpi[i-1]
                if(j == 3): ###(Mean action is East)
                    if((i==4) or (i==9) or (i==14) or (i==19) or (i==24)):
                        q_pi_sa[i,j] = rsa[i,j] + gamma*Vpi[i]
                    else:    
                        q_pi_sa[i,j] = rsa[i,j] + gamma*Vpi[i+1]

    return q_pi_sa


def Pi_iteration(rsa, pssa, gamma, policy, Vpi):
    nstates=rsa.shape[0]
    nactions=rsa.shape[1]
    delta = 0
    theta = 1e-4
    v_current = Vpi
    current_policy = policy


    new_policy = np.zeros((nstates,nactions), dtype="float32")          # 25*4 each is 0.25
    q_pi_sa_of_state = q_pi_sa(rsa,gamma,current_policy,v_current)
    
    
    #print(q_pi_sa_of_state)    
    #print(q_pi_sa_of_state)
    
    for i in range(nstates):
        for j in range(nactions):
            #print(q_pi_sa_of_state[i,j])
            #print(np.max(q_pi_sa_of_state[i]))
            #input("")
            if(q_pi_sa_of_state[i,j] == np.max(q_pi_sa_of_state[i])):
                new_policy[i,j] = 1.0
    

    for i in range(nstates):
        total = np.sum(new_policy[i])
        for j in range(nactions):
            new_policy[i,j] = new_policy[i,j] / total

    #print(new_policy)
    #input("")

    ### Computing the V_pi for new_lopicy
    v_new = veval_matrix(pssa,rsa,new_policy,gamma)
    

    ### update delta
    
    delta = np.max(np.abs(v_current-v_new))
    
    while(delta > theta):
        v_current = v_new
        current_policy = new_policy
        q_pi_sa_of_state = q_pi_sa(rsa,gamma,current_policy,v_current)
        
        
        
        for i in range(nstates):
            for j in range(nactions):
                #print(q_pi_sa_of_state[i,j])
                #print(np.max(q_pi_sa_of_state[i]))
                #input("")
                if(q_pi_sa_of_state[i,j] == np.max(q_pi_sa_of_state[i])):
                    new_policy[i,j] = 1.0
                else:
                    new_policy[i,j] = 0.0
    
        ### Generating new policy according to act greedly for 1_step look_ahead
        for i in range(nstates):
            total = np.sum(new_policy[i])
            for j in range(nactions):
                new_policy[i,j] = new_policy[i,j] / total
        
        v_new = veval_matrix(pssa,rsa,new_policy,gamma)
        

        #print(v_current)
        #print(v_new)
        delta = np.max(np.abs(v_current-v_new))
    

    return new_policy,v_new


#The reward vector r(s,a)
#
rsa=np.zeros((nstates,nactions),dtype=np.float32)
for i in range(5):
    rsa[i,A.North]=-1.0
for i in range(20,25):
    rsa[i,A.South]=-1.0
for i in range(0,25,5):
    rsa[i,A.West]=-1.0
for i in range(4,25,5):
    rsa[i,A.East]=-1.0
#special transition A->A' (state 1->21)
for i in range(nactions): rsa[1,i]=10.0
#special transition B->B' (state 3->13)
for i in range(nactions): rsa[3,i]=5.0


#state-action transition table p(s',s,a)
#
pssa=np.zeros((nstates,nstates,nactions),dtype=np.float32)

#move-north pane
for i in range(nstates):
    if i in [1,3]:
        #special A, B cells
        pass
    elif i in range(5):
        pssa[i,i,A.North]=1.0
    else:
        pssa[i,i-5,A.North]=1.0
        
#move-south pane
for i in range(nstates):
    if i in [1,3]:
        #special A, B cells
        pass
    elif i in range(20,25):
        pssa[i,i,A.South]=1.0
    else:
        pssa[i,i+5,A.South]=1.0
        
#move-west pane
for i in range(nstates):
    if i in [1,3]:
        #special A, B cells
        pass
    elif i in range(0,25,5):
        pssa[i,i,A.West]=1.0
    else:
        pssa[i,i-1,A.West]=1.0
        
#move-east pane
for i in range(nstates):
    if i in [1,3]:
        #special A, B cells
        pass
    elif i in range(4,25,5):
        pssa[i,i,A.East]=1.0
    else:
        pssa[i,i+1,A.East]=1.0
        
#special A, B cells
for move in range(nactions):
    pssa[1,21,move]=1.0
    pssa[3,13,move]=1.0

#
# End Gridworld model creation
#



##########
# Part I  iteration method for computing Vpi function
##########

#Policy function for uniform random policy
policy=np.zeros((nstates,nactions),dtype=np.float32)
policy.fill(1.0/nactions) #4 directions, 25% probability each direction
#solve for value function with iteration method
v=np.sum(policy*rsa,axis=1)
v.shape=(nstates,1) #reshape into column vector first state of iter == 1
v_next =val_iter_syn(v,pssa,rsa,policy,gamma)

print("PART I")
print('Value function computed with iteration method:')
v_next.shape=(5,5) #reshape value vector to match 2-D gridworld shape
print(v_next)



print("----------------------------------------------------------")
print("----------------------------------------------------------")


##########
# Part II  iteration method for computing Pi function
##########

### We start our iteration as following steps
### Step1. A policy chosen at the beginning :  "policy"
### Step2. Computing its Vpi :  "v_next"
### Step3. Generate a better(or equal) policy Pi' by acting greedly on Vpi
### step4. Iteration

### Since we have our first policy as above "policy" and the value function is "v_next"

print("")
print("PART II")

v_next.shape=(25,1)
v_opt = np.zeros((nstates,nactions),dtype='float32')
opt_policy,v_opt = Pi_iteration(rsa, pssa, gamma, policy, v_next)

print('Value function for optimal Pi function:')
v_opt.shape=(5,5) #reshape value vector to match 2-D gridworld shape
print(v_opt)

print("")
print('Optimal Pi function is shown as below:')
print('North South West East')
print(' Up   down  left right')
print(opt_policy)
