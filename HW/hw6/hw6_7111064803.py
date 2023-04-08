#################################################
#                                               #
#                                               #
#                HW6: Sarsa-epsilon             #
#                                               #
#                                               #
#################################################
#
#
## ******************************************** ##
##                                              ##
##  Here only use hw6_7111064803.py             ##
##  and cat_and_mouse.py for running the code   ##
##  If don't want to estimate these states      ##
##  remember to comment policy2gif func.        ##
##                                              ##
## ******************************************** ##

from random import Random
import numpy as np
from cat_and_mouse import Cat_and_Mouse
import matplotlib.pyplot as plt



###
# make things that relate to epsilon a CLASS to make it easier for coding. 
###

class epsilon:
    def __init__(self,epsilon,action):
        self.epsilon_0 = epsilon
        self.epsilon = epsilon
        self.action = list(action)
        self.num_action = len(action)
        self.prng = Random(789)
        self.greedy = 1.0 - epsilon + epsilon/self.num_action
        self.others = epsilon/self.num_action
    
    def choose(self,q,st):
        policy = [self.others]*self.num_action
        greedy_A = np.argmax(q[st])
        policy[greedy_A] = self.greedy
        actions = self.prng.choices(self.action,weights=policy)[0]
        return actions
    
    def epsilon_decay(self,num_episode,num_episodes):
        self.epsilon = self.epsilon_0*(num_episodes-num_episode)/num_episodes
        self.greedy = 1.0 - self.epsilon + self.epsilon/self.num_action
        self.others = self.epsilon/self.num_action

    def policy(self, q, state):
        policy=np.zeros((self.num_action,),dtype=np.float32)
        policy.fill(self.others)
        greedy_A=np.argmax(q[state])
        policy[greedy_A]=self.greedy
        return policy


#####
# Create sarsa function
#####

def sarsa(cat_and_mouse_env,init_st, num_episodes, max_iteration, gamma, alpha):
    
    prng = Random()
    prng.seed(789)

    ### initial q function
    q = np.zeros((cat_and_mouse_env.numStates, cat_and_mouse_env.numActions), dtype=np.float32)

    ### Storing action list according to the current envs
    actions = list(range(cat_and_mouse_env.numActions))

    ### initially setting epsilon between[0,1] but here I just set to 1.0
    eps = 1.0
    ep = epsilon(eps,actions)

    episode = []

    #### LooP
    for num_episode in range(num_episodes):
        #### building a St, At, Rt, first-visit 
        

        current_state = init_st
        cat_and_mouse_env.initState(current_state)
        ep.epsilon_decay(num_episode,num_episodes)
        length = 0
        
        while length < max_iteration:
            ## at St do At get S_nt & reward & game_over or not.
            Action_t = ep.choose(q,current_state)
            (st_n,r,status) = cat_and_mouse_env.step(Action_t)

            q[current_state,Action_t]=q[current_state,Action_t]+alpha*(r+gamma*np.sum(ep.policy(q,st_n)*q[st_n])-q[current_state,Action_t])

            if status: break

            current_state = st_n
            length = length + 1
        episode.append(length)
    
    plt.plot(episode)
    plt.xlabel('Episode')
    plt.ylabel('Length')
    plt.show()
    #cleanup plots
    plt.cla()
    plt.close('all')
    policy=np.argmax(q,axis=1)
    policy.shape=(cat_and_mouse_env.numStates,1)

        
    return (policy,q)



cm=Cat_and_Mouse(rows=1,columns=7,mouseInitLoc=[0,3], cheeseLocs=[[0,0],[0,6]],stickyLocs=[[0,2]],slipperyLocs=[[0,4]])

(policy,q) = sarsa(cat_and_mouse_env=cm,init_st=cm.currentState(),max_iteration=100,num_episodes=500,gamma=0.9,alpha=0.1)
print('1-D problem of cat and mouse result:')
print('optimal policy of q(s,a:)')
print(q)
print('optimal policy function:')
print(policy)

total=cm.policy2gif(policy,[0,3],'cm1d_start_at_03')
print('Total steps starting from {}: {}'.format([0,3],total))
total=cm.policy2gif(policy,[0,2],'cm1d_start_at_02')
print('Total steps starting from {}: {}'.format([0,2],total))




cm_2=Cat_and_Mouse(rows=5,columns=5,mouseInitLoc=[0,0],catLocs=[[3,2],[3,3]], cheeseLocs=[[4,4]],stickyLocs=[[2,4],[3,4]],slipperyLocs=[[1,1],[2,1]])
(policy_2,q_2) = sarsa(cat_and_mouse_env=cm_2,init_st=cm_2.currentState(),max_iteration=250,num_episodes=1800,gamma=0.9,alpha=0.4)
print('2-D problem of cat and mouse result:')
print('optimal policy of q(s,a:)')
print(q_2)
print('optimal policy function:')
print(policy_2)
total=cm_2.policy2gif(policy_2,[0,0],'cm2d_start_at_00')
print('Total steps starting from {}: {}'.format([0,0],total))
total=cm_2.policy2gif(policy_2,[0,3],'cm2d_start_at_03')
print('Total steps starting from {}: {}'.format([0,0],total))
total=cm_2.policy2gif(policy_2,[3,0],'cm2d_start_at_30')
print('Total steps starting from {}: {}'.format([0,0],total))
total=cm_2.policy2gif(policy_2,[1,3],'cm2d_start_at_13')
print('Total steps starting from {}: {}'.format([0,0],total))







