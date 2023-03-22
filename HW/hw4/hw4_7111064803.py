#################################################
#                                               #
#                                               #
#           HW4: Monte Carlo control            #
#                                               #
#                                               #
#################################################
#
#
## ******************************************** ##
##                                              ##
##  Here only use hw4_7111064803.py             ##
##  and cat_and_mouse.py for running the code   ##
##                                              ##
## ******************************************** ##

from random import Random
import numpy as np
from cat_and_mouse import Cat_and_Mouse


#### For each epsiod, we need Q(s,a) to decide for next policy

#####
# Create q(s,a) function
#####

def q_st_at(cat_and_mouse_env, num_episodes, max_iteration, gamma):
    
    prng = Random()
    prng.seed(789)

    ### initial q function
    q = np.zeros((cat_and_mouse_env.numStates, cat_and_mouse_env.numActions), dtype=np.float32)


    ### recording the first visit for (st, at) for all epsiodes
    first_visit = np.zeros((cat_and_mouse_env.numStates, cat_and_mouse_env.numActions), dtype=np.int32)
    
    ### initial policy
    policy = np.zeros((cat_and_mouse_env.numStates, 1),dtype=np.int32)
    for i in range(cat_and_mouse_env.numStates):
        policy[i] = prng.randint(0, cat_and_mouse_env.numActions -1)

    ### Storing action list according to the current envs
    actions = list(range(cat_and_mouse_env.numActions))
    #### LooP
    for num_episode in range(num_episodes):
        #### building a St, At, Rt, first-visit 
        episode = []

        current_state = prng.randint(0, cat_and_mouse_env.numStates-1)
        cat_and_mouse_env.initState(current_state)
        Action_t = prng.choice(actions)
        length = 0
        seen_or_not = np.zeros((cat_and_mouse_env.numStates,cat_and_mouse_env.numActions), dtype=bool) ### using boolean true/false
        while length < max_iteration:
            ## at St do At get S_nt & reward & game_over or not.
            (S_nt, reward, game_end) = cat_and_mouse_env.step(Action_t)
            visit_1st_time = False

            if not seen_or_not[current_state, Action_t]:
                visit_1st_time = True
                seen_or_not[current_state, Action_t] = True
            episode.append((current_state, Action_t, reward, visit_1st_time))

            if(game_end):
                break
            current_state = S_nt
            Action_t = policy[current_state][0]
            length = length + 1
        return_R = 0
        for thing in reversed(episode):     ### reverse calculating
            (S_t, Action_t, reward, visit_1st_time) = thing
            return_R = return_R*gamma + reward

            if(visit_1st_time):
                first_visit[S_t,Action_t] = first_visit[S_t,Action_t] + 1
                ###incremental
                q[S_t, Action_t] = q[S_t,Action_t] + (1.0/first_visit[S_t,Action_t])*(return_R-q[S_t,Action_t])
                ### update policy
                policy[S_t] = np.argmax(q[S_t])
        
    return (policy,q)

cm=Cat_and_Mouse(rows=1,columns=7,mouseInitLoc=[0,3], cheeseLocs=[[0,0],[0,6]],stickyLocs=[[0,2]],slipperyLocs=[[0,4]])

(policy,q) = q_st_at(cat_and_mouse_env=cm,max_iteration=100,num_episodes=5000,gamma=0.9)
print('1-D problem of cat and mouse result:')
print('optimal policy of q(s,a:)')
print(q)
print('optimal policy function:')
print(policy)


cm_2=Cat_and_Mouse(rows=5,columns=5,mouseInitLoc=[0,0],catLocs=[[3,2],[3,3]], cheeseLocs=[[4,4]],stickyLocs=[[2,4],[3,4]],slipperyLocs=[[1,1],[2,1]])
(policy,q) = q_st_at(cat_and_mouse_env=cm_2,max_iteration=100,num_episodes=30000,gamma=0.9)
print('2-D problem of cat and mouse result:')
print('optimal policy of q(s,a:)')
print(q)
print('optimal policy function:')
print(policy)








