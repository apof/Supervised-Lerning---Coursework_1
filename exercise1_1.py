# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 14:11:04 2020

@author: aleks
"""

import numpy as np 
import matplotlib.pyplot as plt

# In[] Exercise 1

# Form training points (x,y)
x = np.transpose(np.array([1.0, 2.0, 3.0, 4.0]))
y = np.transpose(np.array([3.0, 2.0, 0.0, 5.0]))
m=len(x)

# Start the plot which will contain all the fitted curves and training points
t = np.linspace(0, 5, 100)
fig, ax = plt.subplots(figsize=[15,10]) 
ax.scatter(x, y, s=50)

# Fitting curves to data using polynomial bases of dimension k=1,..4

for k in range(1,5):
    
    # Feature map computation using training data points
    phi=np.zeros((m,k)) 
    for i in range(k):
        phi[:,i]=np.power(x,i)
        
    
    # Computing inverse feature map and weights using appropriate math equation
    phi_t=np.transpose(phi)
    phi_1 = np.linalg.inv(np.dot(phi_t,phi)) 
    wk=np.dot(np.dot(phi_1,phi_t),y)
    
    
    # Forming the equation coresponding to the curve fitted and forming the curve to be plotted
    func=np.zeros(len(t))
    equation='k = '+str(i+1)+' => '
    for i in range(k):
        func=func+np.power(t,i)*wk[i]
        equation+=str(round(wk[i],2))+'*x^'+str(i)+'+'
        
        
    # Plotting fitted curves 
    ax.plot(t,func,label='k = '+str(k))   
     
      
    # Calculating MSE error for dimension k
    error=np.dot(phi,wk)-y
    trans_error=np.transpose(error)
    MSE=(np.dot(trans_error,error))/m
    
    
    # Printing the equation coresponding to the curve fitted for dimension k with appropriate MSE
    equation+=' MSE='+str(MSE)
    print(equation)    
        

   
    
leg = ax.legend();    
plt.title('Exercise 1 - fitted polynomial curves')
plt.xlabel('x')
plt.ylabel('y')
plt.xlim([0,5])
plt.ylim([-5,8])
plt.grid()
plt.savefig('exercise1_1_1.png')
plt.show()






# In[] Exercise 2

def generate(x,sigma):
    # Function that generates data set which is made by applying Gsigma function  on
    # input variables x using standard deviation sigma
    
    epsilon = np.random.normal(0,sigma,len(x))
    g=np.square((np.sin(2*np.math.pi*x)))+epsilon
    
    return g


# Generating training data set S of 30 points 
x = np.transpose(np.random.uniform(0,1,30))
m= len(x)
y = generate(x,0.07)

# Generate the test data set T
x_test = np.transpose(np.random.uniform(0,1,1000))
m1=len(x_test)
y_test = generate(x_test,0.07)


# Ploting sin^2(2pit) function and scattered noisy points of function Gsigma
t = np.linspace(0, 1, 1000)
fig, ax = plt.subplots(figsize=[15,10])
func=np.square((np.sin(2*np.math.pi*t)))
ax.plot(t,func,label='Function without the noise')
ax.scatter(x, y, s=50)
leg = ax.legend();
plt.title('Exercise 2-clear function and noise points')
plt.xlabel('x')
plt.ylabel('G')
plt.xlim([0,1.1])
plt.ylim([-0.1,1.1])
plt.grid()
plt.savefig('exercise1_1_2.png')
plt.show()  

# In[]

# Start the plot which will contain all the fitted curves and training points
t = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=[15,10])
ax.scatter(x, y, s=50)
     

# Initialize the MSE value for train and data sets over all k dimensions
MSE=np.zeros(5)
MSE_test=np.zeros(5)
num=0
num1=0

# Fitting curves to data using polynomial bases of dimensions k=2,5,10,14,18

for k in [2,5,10,14,18]:
    
    ## TRAINING
    
    # Feature map computation using training data set S
    phi=np.zeros((m,k)) 
    for i in range(k):
        phi[:,i]=np.power(x,i)
        
    
    # Computing inverse feature map and weights using appropriate math equation
    phi_t=np.transpose(phi)
    phi_1 = np.linalg.inv(np.dot(phi_t,phi)) 
    wk=np.dot(np.dot(phi_1,phi_t),y)
 
    # Forming the curve to be plotted
    func=np.zeros(len(t))
    for i in range(k):
        func=func+np.power(t,i)*wk[i]
        
    # Plotting fitted curves 
    ax.plot(t,func,label='k = '+str(k)) 
    
    # Calculating MSE error for dimension k
    error=np.dot(phi,wk)-y
    trans_error=np.transpose(error)
    MSE[num]=np.dot(trans_error,error)/m
    num+=1
  
    
    ## TESTING
    
    # Feature map computation using test data set T
    phi_test=np.zeros((m1,k))
    for i in range(k):
        phi_test[:,i]=np.power(x_test,i)
    
    # Calculating test MSE error for dimension k
    error_test=np.dot(phi_test,wk)-y_test
    trans_error=np.transpose(error_test)
    MSE_test[num1]=np.dot(trans_error,error_test)/m1
    num1+=1
      
leg = ax.legend(); 
plt.grid()   
plt.xlabel('x')
plt.ylabel('y')
plt.title('Exercise 2 - fitted polynomial curves')    
plt.savefig('exercise1_1_3.png')
plt.show() 

# ln(MSE) plot for the train and test set over dimensions k
fig, ax = plt.subplots(figsize=[15,10])
ax.plot([2,5,10,14,18],np.log(MSE),label='Training')
ax.plot([2,5,10,14,18],np.log(MSE_test),label='Test')
plt.grid()
leg = ax.legend();
plt.xlabel('k')
plt.ylabel('ln(Mean square error)')
plt.title('MSE')
plt.savefig('exercise1_1_3.png')
plt.show() 

# In[] 100 runs

# Initialize the MSE value for train and data sets over all k dimensions and 100 runs
MSE_100=np.zeros(5)
MSE_test_100=np.zeros(5)

for runs in range(100):
    
    # Generating training data set S of 30 points 
    x = np.transpose(np.random.uniform(0,1,30))
    m= len(x)
    y = generate(x,0.07)
    
    # Generate the test data set T
    x_test = np.transpose(np.random.uniform(0,1,1000))
    m1=len(x_test)
    y_test = generate(x_test,0.07)

    
    # Initialize the MSE value for train and data sets over all k dimensions
    MSE=np.zeros(5)
    MSE_test=np.zeros(5)
    num=0
    num1=0
    
    # Fitting curves to data using polynomial bases of dimensions k=2,5,10,14,18
    
    for k in [2,5,10,14,18]:
        
        ## TRAINING
        
        # Feature map computation using training data set S
        phi=np.zeros((m,k)) 
        for i in range(k):
            phi[:,i]=np.power(x,i)
            
        
        # Computing inverse feature map and weights using appropriate math equation
        phi_t=np.transpose(phi)
        phi_1 = np.linalg.inv(np.dot(phi_t,phi)) 
        wk=np.dot(np.dot(phi_1,phi_t),y)
     
        
        # Calculating MSE error for dimension k
        error=np.dot(phi,wk)-y
        trans_error=np.transpose(error)
        MSE[num]=np.dot(trans_error,error)/m
        num+=1
      
        
        ## TESTING
        
        # Feature map computation using test data set T
        phi_test=np.zeros((m1,k))
        for i in range(k):
            phi_test[:,i]=np.power(x_test,i)
        
        # Calculating test MSE error for dimension k
        error_test=np.dot(phi_test,wk)-y_test
        trans_error=np.transpose(error_test)
        MSE_test[num1]=np.dot(trans_error,error_test)/m1
        num1+=1
        
    MSE_100+=MSE
    MSE_test_100+=MSE_test

# Calculating average MSE over 100 runs
MSE_100/=100
MSE_test_100/=100

# ln(avg(MSE)) plot for the train and test set over dimensions k and 100 runs
fig, ax = plt.subplots(figsize=[15,10])
ax.plot([2,5,10,14,18],np.log(MSE_100),label='Training')
ax.plot([2,5,10,14,18],np.log(MSE_test_100),label='Test')
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
plt.grid()
leg = ax.legend();
plt.xlabel('k')
plt.ylabel('ln(MSE)')
#plt.yscale('log')
plt.title('MSE')
plt.savefig('exercise1_1_4.png')
plt.show() 

# In[] Exercise 3


# Initialize the MSE value for train and data sets over all k dimensions and 100 runs
rang=np.array(range(1,19))
MSE_100=np.zeros(len(rang))
MSE_test_100=np.zeros(len(rang))

for runs in range(100):
    
    # Generating training data set S of 30 points 
    x = np.transpose(np.random.uniform(0,1,30))
    m= len(x)
    y = generate(x,0.07)
    
    # Generate the test data set T
    x_test = np.transpose(np.random.uniform(0,1,1000))
    m1=len(x_test)
    y_test = generate(x_test,0.07)

    # Initialize the MSE value for train and data sets over all k dimensions
    MSE=np.zeros(len(rang))
    MSE_test=np.zeros(len(rang))
    num=0
    num1=0
    
    for k in rang:
        
        ## TRAINING
        
        # Feature map computation using training data points and sin functions
        phi=np.zeros((m,k)) 
        for i in range(k):
            phi[:,i]=np.sin((i+1)*np.math.pi*x)
            
        
        # Computing inverse feature map and weights using appropriate math equation
        phi_t=np.transpose(phi)
        phi_1 = np.linalg.inv(np.dot(phi_t,phi)) 
        wk=np.dot(np.dot(phi_1,phi_t),y)
       
            
    
        # Calculating MSE error for dimension k
        error=np.dot(phi,wk)-y
        trans_error=np.transpose(error)
        MSE[num]=np.dot(trans_error,error)/m
        num+=1
      
        ## TESTING
        
        # Feature map computation using test data set T
        phi_test=np.zeros((m1,k))
        for i in range(k):
            phi_test[:,i]=np.sin(i*np.math.pi*x_test)
        
        # Calculating test MSE error for dimension k
        error_test=np.dot(phi_test,wk)-y_test
        trans_error=np.transpose(error_test)
        MSE_test[num1]=np.dot(trans_error,error_test)/m1
        num1+=1
    
    MSE_100+=MSE
    MSE_test_100+=MSE_test


# Calculating average MSE over 100 runs
MSE_100/=100
MSE_test_100/=100

# ln(avg(MSE)) plot for the train and test set over dimensions k and 100 runs
fig, ax = plt.subplots(figsize=[15,10])
ax.plot(rang,np.log(MSE_100),label='Training')
ax.plot(rang,np.log(MSE_test_100),label='Test')
ax.xaxis.set_major_locator(plt.MultipleLocator(2))
plt.grid()
leg = ax.legend();
plt.xlabel('k')
plt.ylabel('ln(MSE)')
#plt.yscale('log')
plt.title('MSE')
plt.savefig('exercise1_1_5.png')
plt.show() 

  
