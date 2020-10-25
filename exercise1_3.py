# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 12:18:25 2020

@author: aleks
"""

import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# In[] Load data and split

dataset_file = 'C:/Users/aleks/Documents/UCL ML/Supervised/Assignment 1/BostonDataset.csv'

## load the dataset into a pandas dataframe
data_frame = pd.read_csv(dataset_file)

# convert dataframe to np array
data_array = data_frame.to_numpy()
print(data_array.shape)

# number of runs for each experiment (defined on the assignement paper)
N = 20
N_fold = 5
features_number = 13
test_split = round(((len(data_array)/3)/len(data_array)*100)/100,2)

train_data, test_data = train_test_split(data_array,test_size=test_split)


x_train = train_data[:,:features_number-1].T
x_test = test_data[:,:features_number-1].T


y_train = np.reshape(train_data[:,(features_number-1)],(1,x_train.shape[1]))
y_test = np.reshape(test_data[:,(features_number-1)],(1,x_test.shape[1]))

# Values for parameters

gamma_values=np.flip(np.power(0.5,np.array(range(26,41))))
sigma_values=np.power(2,np.arange(7.0, 13.5, 0.5))

fold_size= (x_train.shape[1])//N_fold

# In[]
best_gamma=-1
best_sigma=-1
least_val_MSE=-1

avg_val_MSE_plot=[]

                          
for gamma in gamma_values:
    
    avg_val_MSE_plot_help=[]
    
    for sigma in sigma_values:
   
        
        avg_val_MSE=0
        
        print('gamma = '+str(gamma)+', sigma = '+str(sigma))
        
        for fold_id in range(N_fold): # work on first N fold
        
            # N folds
            x_val_cross = x_train[:,fold_id*fold_size:min((fold_id+1)*fold_size,x_train.shape[1])]
            x_train_cross = np.c_[x_train[:,0:fold_id*fold_size],x_train[:,(fold_id+1)*fold_size:x_train.shape[1]]]
            
            y_val_cross = y_train[:,fold_id*fold_size:min((fold_id+1)*fold_size,y_train.shape[1])]
            y_train_cross = np.c_[y_train[:,0:fold_id*fold_size],y_train[:,(fold_id+1)*fold_size:y_train.shape[1]]]            
    
            # Calculate K for this sigma and train folds
            l=x_train_cross.shape[1]
            K_train=np.zeros((l,l)) # K(xi,xj)
            
            
            for i in range(l):
                for j in range(l):
                    K_train[i,j]=np.exp(-np.sum(np.square(x_train_cross[:,i]-x_train_cross[:,j]))/(2*sigma**2))
            
            
            # Calculate K for this sigma and val fold, not square!
            l1=x_val_cross.shape[1]
            K_val=np.zeros((l,l1)) # K(xi,xj)
            
            for i in range(l):
                for j in range(l1):
                    K_val[i,j]=np.exp(-np.sum(np.square(x_train_cross[:,i]-x_val_cross[:,j]))/(2*sigma**2))          
          
                
            # TRAINING for train folds, get alpha* for this combination of gamma and sigma
            
            alpha=np.dot(np.linalg.inv(K_train+gamma*l*np.eye(l)),np.transpose(y_train_cross))
                
            
            # VALIDATION on val fold for this combo, find MSE with Kval, and with Ktrain
            error=np.dot(np.transpose(alpha),K_val)-y_val_cross
            val_MSE=np.dot(error,np.transpose(error))/l
            
             # Average MSE val
            
            avg_val_MSE+= val_MSE
           
            
            
        # Avg MSE and new best paramenters
        avg_val_MSE/=N_fold
        avg_val_MSE_plot_help.append(avg_val_MSE[0,0])
        print('MSE = '+str(avg_val_MSE[0,0]))
       
        if least_val_MSE==-1 or least_val_MSE>avg_val_MSE:
           
            least_val_MSE=avg_val_MSE
            best_gamma=gamma
            best_sigma=sigma

    avg_val_MSE_plot.append(avg_val_MSE_plot_help) # add to a MSE val plot for all combo
    
    
# In[] Plot avg MSE

Z=np.asarray(avg_val_MSE_plot)
X, Y = np.meshgrid(gamma_values, sigma_values)

fig = plt.figure()
ax = plt.axes(projection="3d")

ax.plot_surface(X, Y, Z.T, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')
ax.set_title('MSE val')
plt.show()


# In[]  Retrain on whole train set for the best parameters


# Calculate K for best param
l3=x_train.shape[1]
K_train=np.zeros((l3,l3))
for i in range(l3):
    for j in range(l3):
        K_train[i,j]=np.exp(-np.sum(np.square(x_train[:,i]-x_train[:,j]))/(2*best_gamma**2))
        
l4=x_test.shape[1]
K_test=np.zeros((l3,l4))
for i in range(l3):
    for j in range(l4):
        K_train[i,j]=np.exp(-np.sum(np.square(x_train[:,i]-x_test[:,j]))/(2*best_gamma**2))
        
# Train

alpha=np.dot(np.linalg.inv(K_train+best_gamma*l3*np.eye(l3)),np.transpose(y_train))
     
# Calculate MSE train and test for best parameters

error=np.dot(np.transpose(alpha),K_train)-y_train
train_MSE=np.dot(error,np.transpose(error))/l3

print('Train MSE = '+str(train_MSE[0,0]))

error=np.dot(np.transpose(alpha),K_test)-y_test
val_MSE=np.dot(error,np.transpose(error))/l3
print('Val MSE = '+str(val_MSE[0,0]))

# Do this 20 times and find mean and std MSE for best param, for different train and test splits
         
                
            














