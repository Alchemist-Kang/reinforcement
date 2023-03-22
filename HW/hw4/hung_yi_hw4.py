from random import Random
import numpy as np

def cam_q (map,times_epi,maxlen_move,random_seed=123):

    prng=Random()
    prng.seed(random_seed)

    q=np.zeros((map.numStates,map.numActions),dtype=np.float32)

    
    plcy=np.zeros((map.numStates,1),dtype=np.int32)
    for i in range(map.numStates):
        plcy[i]=prng.randint(0,map.numStates-1)

    tot_visit= np.zeros((map.numStates, map.numActions), dtype=np.int32)
    action_lis = list(range(map.numActions))


    for t in range(times_epi):#迭代次數

        result_list=[]
        
        current_state=prng.randint(0, map.numStates-1)
        current_action=prng.choice(action_lis)

        map.initState(current_state)
        tot_move=0

        visited=np.zeros((map.numStates,map.numActions),dtype=bool)#每次迭代初始化visited,檢查是否來過
        while tot_move < maxlen_move:#不能超過最大移動次數
            print(current_action)
            print(map.step(current_action))
            input("")
            (next_state, reward, term_status) = map.step(current_action)

            first_visit=False
            
            if visited[current_state,current_action] !=True:
                first_visit=True
            visited[current_state,current_action] =True

            result_list.append((current_state, current_action, reward, first_visit))

            if term_status:
                break

            current_state = next_state
            current_action = plcy[current_state][0]  # policy action for next step
            tot_move += 1
        
        tot_rw=0

        for result in reversed(result_list):

            (st, current_action, reward, first_visit) =result

            tot_rw=tot_rw*0.9+reward

            if first_visit == True:
                tot_visit[st,current_action]+=1

                q[st, current_action] = q[st, current_action]+(1.0/tot_visit[st, current_action])*(tot_rw-q[st, current_action])

                plcy[st] = np.argmax(q[st])

    return (plcy, q)