#!/usr/bin/env python
# coding: utf-8
#Author: Adithyan Unni (https://github.com/justadithyan)
#Date: 2nd May 2022 

import numpy as np
import pandas as pd
import os
import math
import matplotlib.pyplot as plt
import scipy.spatial
from random import sample
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [9, 7]

SIN=lambda x: int(math.sin(x * 3.141592653589 / 180))
COS=lambda x: int(math.cos(x * 3.141592653589 / 180))
 
# To rotate an object

def rotate(array, radians):
    """Rotate an array of points by user-defined number of radians"""
    x_arr = []
    y_arr = []
    for i in range(len(array)):
        xy = tuple(array[i, :])
        xfin, yfin = rotate_origin(xy, radians)
        x_arr.append(xfin)
        y_arr.append(yfin)
    return np.array(x_arr), np.array(y_arr)

def rotate_origin(xy, radians):
    """Rotate a point around (0, 0)"""
    x, y = xy
    xx = x * math.cos(radians) + y * math.sin(radians)
    yy = -x * math.sin(radians) + y * math.cos(radians)

    return xx, yy

def stats_computer(no_fish, source_path, frame_count, deficit, sample_count):
    '''For control: computes Mean NND and IFD for specified number of fish. Performs rotation    transform to align misaligned videos'''
    DATA_LIST = []
    file_list = os.listdir(source_path)
    
    file_list = sample(file_list, sample_count)
    for file_name in file_list:
        ind_npz = np.load(os.path.join(BASE_PATH, file_name))
        if file_name in ['control1_fish0.npz', 'control2_fish0.npz', 'control3_fish0.npz', 'control15_fish0.npz']:
            x_data = ind_npz['X']
            y_data = ind_npz['Y']
            data = np.vstack([x_data[deficit:frame_count], y_data[deficit:frame_count]]).reshape(frame_count-deficit, 2)
            x_data, y_data = rotate(data, 0.087)
            data = np.vstack([x_data, y_data])
        else:
            x_data = ind_npz['X']
            y_data = ind_npz['Y']
            data = np.vstack([x_data[deficit:frame_count], y_data[deficit:frame_count]])

        data = data.reshape(2, frame_count - deficit)
        DATA_LIST.append(data)

    total_data = np.vstack(DATA_LIST)
    
    inf_indices = []
    total_data = total_data.T

    for i in range(len(total_data)):
        if np.isinf(sum(total_data[i,:])) == True or np.isnan(sum(total_data[i,:])) == True:
            inf_indices.append(i)
    total_data = np.delete(total_data, inf_indices, 0)

    stats = []

    for row in total_data:
        
        arr = np.array(row)
        pairs = [np.array([arr[i], arr[i+1]]) for i in range(0,2*no_fish - 1,2)]
        positions = np.vstack(pairs)
        dist_mat = np.zeros((no_fish,no_fish))

        for i in range(len(pairs)):
            for j in range(len(pairs)):
                dist_mat[i, j] = math.dist(pairs[i], pairs[j])
                
        dist_mat = dist_mat[~np.eye(dist_mat.shape[0],dtype=bool)].reshape(dist_mat.shape[0],-1)

        nearest_neighbour_distance = np.mean(np.min(dist_mat, axis = 1))
        mean_pairwise_distance = np.mean(np.mean(dist_mat, axis = 1))
        stats.append([nearest_neighbour_distance, mean_pairwise_distance])
#         print('NND, MPD', [nearest_neighbour_distance, mean_pairwise_distance])
#         print('-----------')
    
    final_data = pd.DataFrame(stats, columns = ['nearest_nbor_dist', 'pairwise_dist'])
    return final_data.mean()

def final_stats_computer(no_fish, source_path, frame_count, deficit):
    '''Given T-Rex data for experiment with n fish, compute mean NND and IFD'''
    DATA_LIST = []
    file_list = os.listdir(source_path)
    
    for file_name in file_list:
        ind_npz = np.load(os.path.join(BASE_PATH, file_name))
        x_data = ind_npz['X']
        y_data = ind_npz['Y']
        data = np.vstack([x_data[deficit:frame_count], y_data[deficit:frame_count]])
        data = data.reshape(2, frame_count - deficit)
        DATA_LIST.append(data)

    total_data = np.vstack(DATA_LIST)
    inf_indices = []
    total_data = total_data.T

    for i in range(len(total_data)):
        if np.isinf(sum(total_data[i,:])) == True:
            inf_indices.append(i)
    total_data = np.delete(total_data, inf_indices, 0)
    
    stats = []
    print(total_data.shape)
    
    for row in total_data:
        
        arr = np.array(row)
        pairs = [np.array([arr[i], arr[i+1]]) for i in range(0,2*no_fish - 1,2)]
        positions = np.vstack(pairs)
        dist_mat = np.zeros((no_fish,no_fish))

        for i in range(len(pairs)):
            for j in range(len(pairs)):
                dist_mat[i, j] = math.dist(pairs[i], pairs[j])
                
        dist_mat = dist_mat[~np.eye(dist_mat.shape[0],dtype=bool)].reshape(dist_mat.shape[0],-1)

        nearest_neighbour_distance = np.mean(np.min(dist_mat, axis = 1))
        mean_pairwise_distance = np.mean(np.mean(dist_mat, axis = 1))
        stats.append([nearest_neighbour_distance, mean_pairwise_distance])
#         print('NND, MPD', [nearest_neighbour_distance, mean_pairwise_distance])
#         print('-----------')
        
    
    final_data = pd.DataFrame(stats, columns = ['nearest_nbor_dist', 'pairwise_dist'])
    return final_data.mean()

'''CONTROL'''
bootstrapped_list = []
for i in range(2, 16):
    avg = []
    for j in range(10):
        N_FISH = i
        BASE_PATH = '/home/adithyanunni/Videos/data/controls'
        FRAME_COUNT = 2600
        DEFICIT = 50

        df = stats_computer(no_fish = N_FISH, source_path = BASE_PATH, frame_count = FRAME_COUNT, deficit = DEFICIT, sample_count=N_FISH)
        avg.append(df.to_numpy())
    stacked_avg = np.vstack(avg)
    bootstrapped_list.append(stacked_avg.mean(axis = 0))
    print('For',i, 'fish:', stacked_avg.mean(axis = 0))
control_values = pd.DataFrame(bootstrapped_list, columns = ['Mean NND', 'Mean PD'], index = np.arange(2, 16))

'''EXPERIMENT'''
results_list = []
for i in [5, 10, 15]:
    N_FISH = i
    BASE_PATH = '/home/adithyanunni/Videos/data/' + str(i) + '_fish'
    FRAME_COUNT = 2600
    DEFICIT = 50

    df = final_stats_computer(no_fish = N_FISH, source_path = BASE_PATH, frame_count = FRAME_COUNT, deficit = DEFICIT)
    results_list.append(df)
results_df = pd.DataFrame(np.vstack(results_list), columns = ['NND', 'PD'])

'''PLOTTING RESULTS'''

plt.figure(dpi = 300)
plt.plot(np.arange(2, 16), control_values['Mean NND'], label = 'Expected Nearest-Neighbour Distance', color = '#1f77b4', marker = 'o')
plt.plot(np.arange(2, 16), control_values['Mean PD'], label = 'Expected Inter-fish Distance', color = 'orange', marker = 'o')
plt.plot([5, 10, 15], results_df['NND'], color = '#1f77b4', label = 'Observed Nearest-Neighbour Distance', marker = 'd')
plt.plot([5, 10, 15], results_df['PD'], color = 'orange', label = 'Observed Inter-fish Distance', marker = 'd')
# plt.scatter(np.arange(2, 16), control_values['Mean NND'], color = '#1f77b4')
# plt.scatter(np.arange(2, 16), control_values['Mean PD'], color = 'orange')

plt.xlim([2, 16])
plt.xticks(np.arange(2, 16, 1))
plt.legend()
plt.title('Mean nearest-neighbour and inter-fish distances')
plt.xlabel('Number of fish')
plt.ylabel('Distance (in cm)')
plt.savefig('/home/adithyanunni/Work/animal_behaviour/neig.jpg', dpi = 200)
plt.show()

'''SAVE RESULTS TO CSV'''
control_values.to_csv('/home/adithyanunni/Work/animal_behaviour/supplementary_data/control_values.csv')
results_df.to_csv('/home/adithyanunni/Work/animal_behaviour/supplementary_data/result_values.csv')
