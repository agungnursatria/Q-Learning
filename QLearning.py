# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 07:41:02 2018
@author: Agung Nursatria - 1301150073 - IF3903
"""

import numpy as np
import random

def possible_direction(row,column):
    direction = ['N','E','S','W']
    if (row == 0):
        direction.remove('N')
    if (row == len(R)-1):
        direction.remove('S')
    if (column == 0):
        direction.remove('W')
    if (column == len(R)-1):
        direction.remove('E')
    return direction

def nextState(row,column, direction):
    if(direction == 'N'):
        row = row -1
    if(direction == 'E'):
        column = column + 1
    if(direction == 'S'):
        row = row + 1
    if(direction == 'W'):
        column = column -1
    return [row,column]

def possible_action_Q(Q,qRow,qColumn,direction):
    valueNextAction = []
    for d in direction:
        [nRow,nColumn] = nextState(qRow,qColumn,d)
        valueNextAction.append([d,Q[nRow,nColumn]])
    return valueNextAction
    
##=========== Learning
# Setting Gamma
Gamma = 0.9
# Initialize matrix R
R = np.loadtxt('Data Tugas 3 RL.txt')
# Initialize matrix Q to zero
Q = np.zeros((10,10), dtype=int)

for episode in range(1,1001):
    # initial state
    [row,column] = [9,0]
    current_state = R[row,column]    
    
    while current_state != 100:
        # selecting one of possible direction
        direction = random.choice(possible_direction(row,column))
        [nextRow,nextColumn] = nextState(row,column,direction)
        
        # get possible action from next state
        valueNextAction = possible_action_Q(Q,nextRow,nextColumn,possible_direction(nextRow,nextColumn))
        
        # get maximum Q value for next state and insert to Q next state
        Q[nextRow,nextColumn] = R[nextRow,nextColumn] + (Gamma * max(valueNextAction, key=lambda x:x[1])[1])
        
        # set next state as the current state
        [row,column] = [nextRow,nextColumn]
        current_state = R[row,column] 
    
##=========== Running
# initial state
[row,column] = [9,0]
direction = []
current_state = R[row,column]   
Totalscore = Q[row,column]
ListReward = [current_state]

while current_state != 100:
    # get possible action direction
    direction = possible_direction(row,column)
    listNextAction = possible_action_Q(Q,row,column,direction) 
    
    # get best direction
    bestDirection = max(listNextAction, key=lambda x:x[1])[0]
    
    # set next state as the current state
    [nextRow,nextColumn] = nextState(row,column,bestDirection)
    [row,column] = [nextRow,nextColumn]
    current_state = R[row,column] 
    Totalscore = Totalscore + Q[row,column]
    ListReward.append(current_state)
    
print('========================================')
print('Reward = ', sum(ListReward))
print('Number of Action = ', len(ListReward)-1)
print('Total Score = ', Totalscore)
print('========================================')