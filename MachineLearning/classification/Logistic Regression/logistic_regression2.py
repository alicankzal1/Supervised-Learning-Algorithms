# importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%

df = pd.read_csv("data.csv")

df.head()
print(df.info())

print(df.isna().sum())

df.drop(["id", "Unnamed: 32"], axis= 1, inplace= True)

## diagnosis columnunu objectten int değere değiştirmek (list comprehension)

df.diagnosis = [1 if each == "M" else 0 for each in df.diagnosis]


# X ve y değerlerine ayırmak

y = df.diagnosis.values
# X diagnosis hariç hepsi olur
X = df.drop("diagnosis", axis= 1)
#%% Normalization 

X = (X - np.min(X)) / (np.max(X) - np.min(X))
#%% train test split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)

X_train = X_train.T
X_test = X_test.T
y_train = y_train.T
y_test = y_test.T

print(f"X_train : {X_train.shape}\nX_test : {X_test.shape}\ny_train : {y_train.shape}\ny_test : {y_test.shape}")
#%% parameter initialize and sigmoid func
# bu datasette dimonsion = 30

def initialize_and_bias(dimension):
    w = np.full((dimension, 1), 0.01) #shape (dimesion, 1) olacak ve bunları 0.01 ile dolduracak
    b = 0.0
    return w,b
#w, b = initialize_and_bias(30)
#print(w, b)

def sigmoid(z):
    y_head = 1 / (1 + np.exp(-z)) # sigmoid function = 1 / (1 + e^(-z)
    return y_head

#print(sigmoid(0)) # sigmoid 0 ise 0.5 döner
#%%
def forward_and_backward_propagation(w, b, X_train, y_train):
    #forward propagation
    z = np.dot(w.T, X_train) + b
    y_head = sigmoid(z)
    loss = -y_train * np.log(y_head) - (1 - y_train) * np.log(1 - y_head)
    cost = (np.sum(loss)) / X_train.shape[1] #X_train.shape[1] is for scaling
    
    #backward propagation
    derivative_weight = (np.dot(X_train,((y_head - y_train).T))) / X_train.shape[1] #w türev alma - X_train.shape[1] is for scaling
    derivative_bias = (np.sum(y_head - y_train)) / X_train.shape[1]
    gradients = {
        "derivative_weight" : derivative_weight,
        "derivative_bias": derivative_bias
        }
    return cost, gradients
    
#%% updating(learning) parameters

def update(w, b, X_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []
    
    for i in range(number_of_iteration):
        
        cost, gradients = forward_and_backward_propagation(w, b, X_train, y_train)
        cost_list.append(cost)
        
        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]
        
        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    parameters = {
        "weight" : w,
        "bias" : b}
    plt.plot(index, cost_list2)
    plt.xticks(index, rotation= "vertical")
    plt.xlabel("number of iteration")
    plt.ylabel("Cost")
    plt.show()
    
    
    
    return parameters, gradients, cost_list
#%% prediction

def predict(w, b, X_test):
    
    z = sigmoid(np.dot(w.T, X_test) + b)
    Y_prediction = np.zeros((1,X_test.shape[1]))
    
    
    for i in range(z.shape[1]):
        
        if z[0,i] <= 0.5:
            Y_prediction[0,i] = 0
        else:
            Y_prediction[0,i] = 1
            
    return Y_prediction
#%% logistic regression

def logistic_regression(X_train, X_test, y_train, y_test, learning_rate, num_iterations):
    #initialize
    dimension = X_train.shape[0]  #that is 30
    
    w, b = initialize_and_bias(dimension)
    
    parameters, gradients, cost_list = update(w, b, X_train, y_train, learning_rate, num_iterations)
    y_prediction_test = predict(parameters["weight"], parameters["bias"], X_test)
    
    #print test errors
    print("test accuracy : % {}".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100,))
    
logistic_regression(X_train, X_test, y_train, y_test, learning_rate= 1, num_iterations= 2000)


#%% Logistic Regression with sklearn

from sklearn.linear_model import LogisticRegression

logistic_reg = LogisticRegression()

logistic_reg.fit(X_train.T, y_train.T)
print(f"test accuracy {logistic_reg.score(X_test.T, y_test.T)}")


























        






