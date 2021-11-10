import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import random
import matplotlib
type_name = 'Atari'
ENVS = ['Breakout','SpaceInvaders','Enduro','CrazyClimber']
env_name = ENVS[0]
sns.set(context='notebook', style='darkgrid', palette='deep', font='sans-serif', font_scale=1, color_codes=False, rc=None)
sns.set_style("whitegrid")
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = 'NSimSun,Times New Roman'

Methods = ['DuelingDQN', 'DuelingDQN_RAA', 'DuelingDQN_StableAA(ours)']


final_x = []
final_y = []
colors=['k', 'blue', 'red', 'c', 'm', 'y', 'w']

# SUBPLOTS = [221, 222, 223,224]
SUBPLOTS = [111]

for k in range(len(SUBPLOTS)):
    plt.subplot(SUBPLOTS[k])
    # (0) env
    env_name = ENVS[0] # k
    # (1) process data
    current_dir = "logs/" + env_name + "NoFrameskip-v4/"
    total_result_x = []
    total_result_y = []
    result_x = []
    type = []  # !!! filename for plot
    for lists in os.listdir(current_dir): # loop in 3 files
        # filename
        current_type = lists  # Dueling, Dueling_RAA, StableAA
        next_dir = current_dir + lists + "/" # one algorithm path
        type.append(current_type)
        current_type_result_x = []
        current_type_result_y = []

        for seed in os.listdir(next_dir): # seed-101, seed-102
            temp_result = np.load(next_dir + seed + "/scalars.npy")
            # get the current result
            temp_result_y = []
            temp_result_x = []
            for a in temp_result:
                temp_result_y.append(a[-2])  # performance
                temp_result_x.append(a[1])  # steps
            current_type_result_x.append(len(temp_result_y))  # add the length of the evaluation steps
            current_type_result_y.append(temp_result_y)
            result_x.append(temp_result_x)  # add the step intervals

        total_result_x.append(current_type_result_x)  # length of steps over 3 seeds
        total_result_y.append(current_type_result_y)

    print('length &&&&&&&&&&&&&&&&')
    print(total_result_x)
    min_length = min(min(row) for row in total_result_x)
    print(min_length)

    # (2) plot
    for i in range(len(total_result_y)): # length of files
        current_type = type[i] # filename
        # identify types
        temps = current_type.split('-')
        temps1 = temps[0]
        temps4 = temps[4]
        if temps1 == 'DuelingDQN':
            index = 0
        elif temps1 == 'DuelingDQN_RAA' and int(temps4)==0:
            index = 1
        else: # stable AA
            index = 2
        temp_y = []
        for j in total_result_y[i]: # total_result_y: N,seeds; j loop in seeds results
            temp_y.append(j[:min_length])
        # sns.tsplot(time=result_x[0][:min_length], data=temp_y, color=colors[i], condition=current_type) # temp_y: 3 lists
        steps = [i/1e7 for i in result_x[0][:min_length]]
        sns.tsplot(time=steps, data=temp_y, color=colors[index], condition=Methods[index]) # temp_y: 3 lists
    plt.yticks(fontproperties='Times New Roman', size=18)
    plt.xticks(fontproperties='Times New Roman', size=18)
    plt.xlabel('Time Steps (1e7)', fontsize=28)
    plt.ylabel('Average Return', fontsize=28)
    plt.title(env_name, fontsize=33)
    plt.legend(loc=2, fontsize=15)
plt.show()



