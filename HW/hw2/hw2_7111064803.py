#################################################
#                                               #
#                                               #
#   HW2: Value Function using matrix approach   #
#                                               #
#                                               #
#################################################

import numpy as np
from numpy.linalg import inv

np.set_printoptions(threshold=np.inf)           ### Numpy array can print infinite (original setting is 1000)

gamma = 0.9
 
cells = np.arange(start=0, stop=25, step=1)
#print(cells, cells.dtype)
Rewards = np.zeros(shape=(25), dtype='float64')
Probability_s_a = np.zeros(shape=(25,25), dtype='float64')
#print(Probability_s_a, Probability_s_a.dtype)

I_matrix = np.identity(Probability_s_a.shape[0])

for i in range(Probability_s_a.shape[0]):                ## i is row
    if(((i>5) and (i<19) and (i != 9) and (i != 10) and (i != 14) and (i != 15))):
        Probability_s_a[i,i-1] = 0.25
        Probability_s_a[i,i+1] = 0.25
        Probability_s_a[i,i-5] = 0.25
        Probability_s_a[i,i+5] = 0.25
        Rewards[i] = 0

    if( (i == 5) or (i == 10) or (i ==15)): 
        Probability_s_a[i,i] = 0.25
        Probability_s_a[i,i+1] = 0.25
        Probability_s_a[i,i-5] = 0.25
        Probability_s_a[i,i+5] =0.25
        Rewards[i] = 0.25*(-1.0)

    if((i == 9) or (i == 14) or (i == 19)):
        Probability_s_a[i,i] = 0.25
        Probability_s_a[i,i-1] = 0.25
        Probability_s_a[i,i-5] = 0.25
        Probability_s_a[i,i+5] = 0.25
        Rewards[i] = 0.25*(-1.0)

    if((i == 21) or (i == 22) or (i == 23)):
        Probability_s_a[i,i] = 0.25
        Probability_s_a[i,i-1] = 0.25
        Probability_s_a[i,i+1] = 0.25
        Probability_s_a[i,i-5] = 0.25
        Rewards[i] = 0.25*(-1.0)

    if(i == 2):
        Probability_s_a[i,i] = 0.25
        Probability_s_a[i,i-1] = 0.25
        Probability_s_a[i,i+1] = 0.25
        Probability_s_a[i,i+5] = 0.25
        Rewards[i] = 0.25*(-1.0)

    if(i == 0): 
        Probability_s_a[i,i] = 0.5
        Probability_s_a[i,+1] = 0.25
        Probability_s_a[i,i+5] = 0.25
        Rewards[i] = 0.5*(-1.0)
    
    if(i == 4):
        Probability_s_a[i,i] = 0.5
        Probability_s_a[i,i-1] = 0.25
        Probability_s_a[i,i+5] = 0.25
        Rewards[i] = 0.5*(-1.0)

    if(i == 20):
        Probability_s_a[i,i] = 0.5
        Probability_s_a[i,i+1] = 0.25
        Probability_s_a[i,i-5] = 0.25
        Rewards[i] = 0.5*(-1.0)

    if(i == 24):
        Probability_s_a[i,i] = 0.5
        Probability_s_a[i,i-1] = 0.25
        Probability_s_a[i,i-5] = 0.25
        Rewards[i] = 0.5*(-1.0)

    if(i == 1):
        Probability_s_a[i,i+20] = 1
        Rewards[i] = 10.0
    
    if(i == 3):
        Probability_s_a[i,i+10] = 1
        Rewards[i] = 5.0
#print(Probability_s_a)
#print(Rewards, Rewards.dtype)


Value_function = np.dot(inv(I_matrix - gamma*(Probability_s_a)), Rewards)
Value_function = np.round(Value_function,decimals=1)

print(Value_function)





