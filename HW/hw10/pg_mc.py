#
# Policy-gradient with Monte Carlo (REINFORCE)
#
#

#
#"Skeleton" file for HW...
#
# Your task: Complete the missing code (see comments below in the code):
#            1. Implement Policy Gradient with Monte Carlo (REINFORCE)
#            2. Test your algorithm using the provided pg_mc_demo.py file
#


import torch 

#
#   Policy-gradient with Monte Carlo
#
def pg_mc(simenv, policy, gamma, alpha, num_episodes, max_episode_len, window_len=100, term_thresh=None, showPlots=False, prng_seed=789):
    '''
    Parameters:
        simenv:  Simulation environment instance
        policy:  Parameterized model for the policy (e.g. neural network)
        gamma :  Future discount factor, between 0 and 1
        alpha :  Learning step size
        num_episodes: Number of episodes to run
        max_episode_len: Maximum allowed length of a single episode
        window_len: Window size for total rewards windowed average (averaged over multiple episodes)
        term_thresh: If windowed average > term_thresh, stop (if None, run until num_episodes)
        showPlots:  If True, show plot of episode lengths during training, default: False
        prng_seed:  Seed for the random number generator        
    '''

    #initialize a few things
    #    
    simenv.reset(seed=prng_seed)

    #
    #You might need to add some additional initialization code here,
    #  depending on your specific implementation
    #
    optim = torch.optim.Adam(policy.parameters(),lr=alpha,maximize=True)    ### Since its gradient ascent
    



    ###########################
    #Start episode loop
    #
    episodeLengths=[]
    episodeRewards=[]
    averagedRewards=[]

    for episode in range(num_episodes):
        if episode%100 == 0:
            print('Episode: {}'.format(episode))

        #initial state
        state=simenv.reset()
        
        #
        #You might need to add some code here,
        #  depending on your specific implementation
        #
        ### For tracking each step's reward && gradient of Pi()
        Rewards = []
        ln_Pis = []

        #Run episode state-action-reward sequence to end
        #
        episode_length=0
        tot_reward=0
        
        while episode_length < max_episode_len:

            #
            #Fill in the missing algorithm code here!
            # (Note: test your results with the pg_mc_demo.py file)
            #
            #pass #delete this!

            ### For each episode, take actions based on policy, get reward from it, than store in the list as [S0,A0,R1 ,..., St-1,At-1,Rt] for further computing
            ### In this way, we could avoid that divide by [Pi() equals to zero]

            ### Use current policy to decide the action
            (action,ln_pi) = policy.choose_action(state)

            ### Take action, get reward, appears in new state & terminated status
            (next_state,reward,term_status,_) = simenv.step(action)
            tot_reward += reward

            Rewards.append(reward)
            ln_Pis.append(ln_pi)

            if term_status: break
            state = next_state
            episode_length += 1

        ### Computes the discounted return Gt
        ### "Iteratively" : Using backward method computes from [St-1 At-1 Rt] to [S0,A0,R1] and tracking the gradients with backward() ----> accumulating
        total_return = 0
        total_return_list = torch.zeros(len(Rewards),dtype=torch.float32)
        for r in range(len(Rewards)-1,-1,-1):
            total_return = gamma*total_return + Rewards[r]
            total_return_list[r] = total_return

        ### Computes gradient target for an episode
        ln_pi_stack = torch.stack(ln_Pis)       ### Stored in a stack to eqsily cumputes with discounted return list [Gt]
        target = torch.sum(total_return_list*ln_pi_stack)

        ### backward() functions to update gradients, step() function to update weights
        target.backward()
        optim.step()
        optim.zero_grad()   ### For next episode clean the gradient accumulation
        




        #update stats for later plotting
        episodeLengths.append(episode_length)
        episodeRewards.append(tot_reward)
        avg_tot_reward=sum(episodeRewards[-window_len:])/window_len
        averagedRewards.append(avg_tot_reward)

        if episode%100 == 0:
            print('\tAvg reward: {}'.format(avg_tot_reward))

        #if termination condition was specified, check it now
        if (term_thresh != None) and (avg_tot_reward >= term_thresh): break


    #if plot metrics was requested, do it now
    if showPlots:
        import matplotlib.pyplot as plt
        plt.subplot(311)
        plt.plot(episodeLengths)
        plt.xlabel('Episode')
        plt.ylabel('Length')
        plt.subplot(312)
        plt.plot(episodeRewards)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.subplot(313)
        plt.plot(averagedRewards)
        plt.xlabel('Episode')
        plt.ylabel('Avg Total Reward')
        plt.show()
        #cleanup plots
        plt.cla()
        plt.close('all')
