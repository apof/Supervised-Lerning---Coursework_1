import numpy as np
from sklearn.model_selection import train_test_split
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error

## define linear regression
def linear_regression_fit(X,Y):
    return inv(X.T@X)@X.T@Y


# (a) Predicting with the mean y-value on the training set
# question 1.2.a
def fit_with_constant_function(features_number,data_array,test_split,N):
    for i in range(0,N):
    
        # perform a new data split on train and test set on every rou
        # suffle = True by default in this sklearn function
        train_data, test_data = train_test_split(data_array,test_size=test_split)

        if(i==0):
            ## initialize on the first round
            print("Initialization!")
            ## initialize x vectors with ones - fit the data with constant function
            x_train = np.asmatrix(np.ones(len(train_data))).T
            x_test = np.asmatrix(np.ones(len(test_data))).T
            y_train = train_data[:,(features_number-1)]
            y_test = test_data[:,(features_number-1)]
            mse_train = np.zeros(N)
            mse_test = np.zeros(N)
    
        weights_train = linear_regression_fit(x_train,y_train)
        #weights_test = linear_regression_fit(x_test,y_test)
    
        mse_train[i]= mean_squared_error(x_train*weights_train, y_train)
        mse_test[i]= mean_squared_error(x_test*weights_train, y_test)
    
    print("Average MSE for train and test set are: " + str(np.mean(mse_train)) + " and " + str(np.mean(mse_test)) + " respectively")


    ## (b) Predicting with a single attribute and a bias term.

def fit_with_one_feature(features_number,data_array,test_split,columns,N):

    mse_train = np.zeros((N,(features_number-1)))
    mse_test = np.zeros((N,(features_number-1)))

    for i in range(0,N):
    
        train_data, test_data = train_test_split(data_array,test_size=test_split)
        
        for j in range(0,(features_number-1)):
        
            y_train = train_data[:,(features_number-1)]
            y_test = test_data[:,(features_number-1)]
        
            ## select one feature on every round
            ## and one extra dimension with 1 values on the x
            x_train = np.asmatrix(np.vstack((train_data[:,j],np.ones(len(train_data))))).T
            x_test = np.asmatrix(np.vstack((test_data [:,j],np.ones(len(test_data))))).T
        
            weights_train = linear_regression_fit(x_train,y_train).T
            #weights_test = linear_regression_fit(x_test,y_test).T
       
            mse_train[i,j]= mean_squared_error(x_train@weights_train, y_train)
            mse_test[i,j]= mean_squared_error(x_test@weights_train, y_test)
        
    print("Linear Regression with single Features - Results")   
    for i in range(len(mse_train.T)):
        print("Linear regression using attribute: " + str(columns[i]) + " MSE train: " + str(np.mean((np.sum(mse_train,axis = 0)/N).T[i])) +" MSE test:" + str(np.mean((np.sum(mse_test,axis = 0 )/N).T[i])))


 ## Predicting with all the features
def fit_with_all_features(features_number,data_array,test_split,N):
    for i in range(N):
    
        mse_train = np.zeros(N)
        mse_test  = np.zeros(N)
    
        train_data, test_data = train_test_split(data_array,test_size=test_split)
     
        ## use all the features
        y_train = train_data[:,(features_number-1)]
        y_test = test_data[:,(features_number-1)]
    
        ## add one dimension with one to x data
        x_train = np.asmatrix(np.c_[(train_data[:,range(features_number)],np.ones(len(train_data)))])
        x_test = np.asmatrix(np.c_[(test_data[:,range(features_number)] ,np.ones(len(test_data )))])
        
        w_train = linear_regression_fit(x_train,y_train).T
        w_test = linear_regression_fit(x_test,y_test).T
        
        mse_train[i]= mean_squared_error(x_train@w_train, y_train)
        mse_test [i]= mean_squared_error(x_test@w_test, y_test)
        
    print("Linear regression using all the features with MSE train: " + str(np.sum(mse_train)/N) + " MSE test: " +  str(np.sum(mse_test )/N))

