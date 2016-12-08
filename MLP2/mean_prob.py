import numpy as np
import random

prob = 0.75

with open("random.csv", mode='w') as f:
    f.write("ID,Prediction\n")
    for i in range(0,138):
        R = random.uniform(0,1)
        if R <= prob:
            pred = 0.75
        else:
            pred = 0.25
        f.write(str(i+1)+','+str(pred)+'\n')

