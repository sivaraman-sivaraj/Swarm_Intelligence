"""
Created on Mon Nov  9 23:40:40 2020

@author: Sivaraman Sivaraj
"""

import Particle_Swarm_Optimization as pso
import numpy as np

N = np.random.randint(-100,100,(50,5))

# def f(x):
#     temp = 5*(x[0]**5) - 3*(x[1]**4) + 6*x[2]
#     temp1 = x[3] - 10*x[4]
#     return temp+temp1

def f(x): #for example, we use this square function
    value = 0
    for i in range(len(x)):
        value += (x[i]**2)
    return value

T = 100 #number of iteration
w = 0.9 # inertia weight
c1 = 1.8 #cognitive term weight
c2 = 2.01 #spcial term weight
lb = 0
ub = 99

G_vector, G_best, N_updated, F_updated,N = pso.Do_PSO(N,T,w,c1,c2,f,lb,ub)

print("The global optimal value after",str(T),"iterations is " , G_best)






