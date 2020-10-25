import numpy as np
from sklearn.model_selection import train_test_split
from numpy.linalg import inv

def mean_squared_error(y_pred,y_true):
	mse = 0
	for i in range(0,len(y_true)):
		mse += (y_true[i]-y_pred[i])**2
	return mse/len(y_true)

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
            y_train = np.asmatrix(train_data[:,(features_number-1)]).T
            y_test = np.asmatrix(test_data[:,(features_number-1)]).T
            mse_train = np.zeros(N)
            mse_test = np.zeros(N)

            #print("Training set shape: " + str(x_train.shape))
            #print(x_test.shape)
            #print("Training labels shape: " + str(y_train.shape))
            #print(y_test.shape)
            #print("----")
    
        weights_train = linear_regression_fit(x_train,y_train)
        #weights_test = linear_regression_fit(x_test,y_test)

        #print("Weights shape: " + str(weights_train.shape))

        training_preds = x_train@weights_train
        test_preds = x_test@weights_train

        #print("Training predictions shape:  " + str(training_preds.shape))
    
        mse_train[i]= mean_squared_error(training_preds,y_train)
        mse_test[i]= mean_squared_error(test_preds,y_test)
    
    print("Average MSE for train and test set are: " + str(np.mean(mse_train)) + " and " + str(np.mean(mse_test)) + " respectively")


    ## (b) Predicting with a single attribute and a bias term.

def fit_with_one_feature(features_number,data_array,test_split,columns,N):

	mse_train = np.zeros((N,(features_number-1)))
	mse_test = np.zeros((N,(features_number-1)))

	for i in range(0,N):
    
		train_data, test_data = train_test_split(data_array,test_size=test_split)
        
		for j in range(0,(features_number-1)):

			#print("Fiting for feature: " + str(columns[j]))
        
			y_train = train_data[:,(features_number-1)].T
			y_test = test_data[:,(features_number-1)].T
        
			## select one feature on every round
			## and one extra column dimension with 1 values on the x
			x_train = np.asmatrix(np.c_[train_data [:,j],np.ones((train_data.shape[0],1))])
			x_test = np.asmatrix(np.c_[test_data [:,j],np.ones((test_data.shape[0],1))])

			#print("Training set shape: " + str(x_train.shape))
			#print(x_test.shape)
			#print("Training labels shape: " + str(y_train.shape))
			#print(y_test.shape)
			#print("----")
        
			weights_train = linear_regression_fit(x_train,y_train)

			#print("Weights shape: " + str(weights_train.shape))
			
			training_preds = x_train@(weights_train.T)
			test_preds = x_test@(weights_train.T)

			#print(np.mean(y_train))
			#print(np.mena(training_preds))

			#print("Training predictions shape:  " + str(training_preds.shape))
      
			mse_train[i,j]= mean_squared_error(training_preds, y_train)
			mse_test[i,j]= mean_squared_error(test_preds , y_test)
        
	print("Linear Regression with single Features - Results")   
	for i in range(len(mse_train.T)):
		print("Linear regression using attribute: " + str(columns[i]) + " MSE train: " + str(np.mean((np.sum(mse_train,axis = 0)/N).T[i])) +" MSE test:" + str(np.mean((np.sum(mse_test,axis = 0 )/N).T[i])))


 ## Predicting with all the features
def fit_with_all_features(features_number,data_array,test_split,N):

	mse_train = np.zeros(N)
	mse_test  = np.zeros(N)

	for i in range(N):

		train_data, test_data = train_test_split(data_array,test_size=test_split)

		y_train = train_data[:,(features_number-1)].T
		y_test = test_data[:,(features_number-1)].T

		## use all the features
		## and one extra column dimension with 1 values on the x
		x_train = np.asmatrix(np.c_[train_data [:,range(features_number-1)],np.ones((train_data.shape[0],1))])
		x_test = np.asmatrix(np.c_[test_data [:,range(features_number-1)],np.ones((test_data.shape[0],1))])


		#print("Training set shape: " + str(x_train.shape))
		#print(x_test.shape)
		#print("Training labels shape: " + str(y_train.shape))
		#print(y_test.shape)
		#print("----")
        
		weights_train = linear_regression_fit(x_train,y_train)

		#print("Weights shape: " + str(weights_train.shape))
			
		training_preds = x_train@(weights_train.T)
		test_preds = x_test@(weights_train.T)

		#print("Training predictions shape:  " + str(training_preds.shape))
        
		mse_train[i] = mean_squared_error(training_preds, y_train)
		mse_test [i] = mean_squared_error(test_preds, y_test)
        
	print("Linear regression using all the features with MSE train: " + str(np.sum(mse_train)/N) + " MSE test: " +  str(np.sum(mse_test )/N))

