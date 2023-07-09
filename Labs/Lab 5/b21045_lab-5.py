import math
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
from sklearn import mixture
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn . mixture import GaussianMixture
from sklearn import metrics
from sklearn.linear_model import LinearRegression
import sklearn.mixture  
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures


X_train = pd.read_csv('SteelPlateFaults-train.csv') # imported from lab 4
X_test = pd.read_csv('SteelPlateFaults-test.csv') # imported from lab 4


del_list = ['TypeOfSteel_A400','TypeOfSteel_A300','X_Minimum','Y_Minimum'] # deleted so that the determinant of covaraiance matrix is not 0

for i in del_list:
    del X_train[i] 
    del X_test[i] 

X_train_class0 = X_train[X_train['Class'] == 0] # creating train data for class  =  0
X_train_class1 = X_train[X_train['Class'] == 1] # creating train data for class  =  1

del X_train_class0['Class'] # deleting class from train data of class  =  0
del X_train_class1['Class'] # deleting class from train data of class  =  1

Actual_Class = X_test.Class # storing the actual class as a list

del X_test['Class'] # deleting class from test data

ll = [2,4,8]
accuracy = [] # storing the accuracies

for i in ll:
    NGMM_comp = i
    GMM1 = mixture.GaussianMixture(n_components = NGMM_comp,covariance_type = 'full') # creating GMM of class 0 
    GMM1.fit(X_train_class0) # fitting test data to GMM of class 0 
    GMM2 = mixture.GaussianMixture(n_components = NGMM_comp,covariance_type = 'full') # creating GMM of class 1 
    GMM2.fit(X_train_class1) # fitting test data to GMM of class 1 

    predicted_class = np.argmax([GMM1.score_samples(X_test),GMM2.score_samples(X_test)],0) # giving the prediction based on calculated probabilities
    acc = metrics.accuracy_score(Actual_Class,predicted_class) # giving the accuracy of the prediction
    print("Classification Accuracy for Q  = ",i ,"is:",acc)
    accuracy.append(acc)
    conf_mat = metrics.confusion_matrix(Actual_Class,predicted_class)
    print ("Confusion Matrix for Q  = ",i ,"is :")
    print(conf_mat)

print("The Maximum accuracy is when the number of components in the GMM   =  4")
 # for l1 = 16 we are getting following error:
 # ValueError: Fitting the mixture model failed because some components have ill-defined empirical covariance (for instance caused by singleton or collapsed samples). 
 # Try to decrease the number of components, or increase reg_covar.

Accuracy = [0.8694362017804155, 0.8961424332344213, 0.8931750741839762, 0.9614243323442137, 0.9703264094955489, 0.973293768545994, 0.9436201780415431] # obtained in lab 4
Accuracy.extend(accuracy) # appends the values of accuracy of the GMM obtained above
print("The maximum accuracy is of KNN Normalization data:")
print(max(Accuracy))

df = pd.read_csv('abalone.csv') # reading the data from the csv
[X_train, X_test] = sklearn.model_selection.train_test_split(df, test_size = 0.3, random_state = 42,shuffle = True) # creating the test and train data
X_train.to_csv('abalone-train.csv') # creating new csv files for the train data
X_test.to_csv('abalone_test.csv') # creating new csv files for the test data

pearson_correl = df.corr(method = 'pearson') # obtaining the pearson correlation values for all the attributes (correlation matrix)
pearson_correl_Rings = pearson_correl["Rings"].to_dict() # getting the correlation of all the attributes with the ring attribute in the form of a dictionary
del pearson_correl_Rings["Rings"] # removing it as this will give the value as 1
max_corr_col = max(pearson_correl_Rings,key = pearson_correl_Rings.get) # obtaining the maximum correlation value of an attribute with the attribute rings

regressor = LinearRegression()
x = X_train[max_corr_col].values.reshape(-1,1) # reshaping the columns values into a single into an array. We use -1 as we dont know the number of rows but we want them sep
y = X_train["Rings"].values.reshape(-1,1) # reshaping the columns values into a single into an array. We use -1 as we dont know the number of rows but we want them sep
regressor.fit(x,y) # fitting the values in x and y into linear regression
y_pred = regressor.predict(x) # using the regression to predict x


# creating scatter-plot
plt.scatter(x,y,label = "Training data",edgecolor = "black")
plt.plot(x,y_pred,color = "red",label = "Best fit line")
plt.xlabel(max_corr_col)
plt.ylabel("Rings")
plt.legend()
plt.show()

RMSE_train_data = mean_squared_error(y, y_pred,squared = False) # calulating the rmse error in the actual y values and the predicted y values
print("The RMSE value for the train data for univariate linear regression =" , RMSE_train_data)

x1 = X_test[max_corr_col].values.reshape(-1,1) # reshaping the list into a list of lists for the test data
y1 = X_test["Rings"].values.reshape(-1,1) # reshaping the list into a list of lists for the test data
y_pred_test = regressor.predict(x1)
RMSE_test_data = mean_squared_error(y1, y_pred_test,squared = False)
print("The RMSE value for the test data for univariate linear regression =",RMSE_test_data)

plt.scatter(y1,y_pred_test,color = "green",edgecolor = "black")
plt.xlabel("Actual Rings")
plt.ylabel("Calculated Rings")
plt.title("scatter plot of actual Rings vs predicted Rings on test data")
plt.show()

#Question2

X_train1=X_train.copy() 
del X_train1["Rings"]
regressor.fit(X_train1,X_train["Rings"]) #Applying multivariate Linear Regression model

y_pred_train=regressor.predict(X_train1) #Predicting the train data using constructed model

RMSE_train_data=mean_squared_error(X_train["Rings"],y_pred_train,squared=False) #Calculating the RMSE
print("Root mean squared error on Train data in multivariate linear regression model ",RMSE_train_data)

X_test1=X_test.copy()
del X_test1["Rings"]
y_pred_test=regressor.predict(X_test1) #Predicting the test data using constructed model

RMSE_test_data=mean_squared_error(X_test["Rings"],y_pred_test,squared=False) #Calculating the RMSE
print("Root mean squared error on Test data in multivariate linear regression model ",RMSE_test_data)

#Ploting the scatter plot of actual Rings vs predicted Rings on test data
plt.scatter(X_test["Rings"],y_pred_test,edgecolor='black')
plt.xlabel("Actual Rings")
plt.ylabel("Calculated Rings")
plt.title("scatter plot of actual Rings vs predicted Rings on test data")
plt.show()

pearson_correl = df.corr(method ='pearson') # obtaining the correlation matrix
pearson_correl_Rings = pearson_correl["Rings"].to_dict() # obtaining the values of correlation with rings in the form of dictionary 
del pearson_correl_Rings["Rings"]
max_col = max(pearson_correl_Rings,key = pearson_correl_Rings.get)

regressor = LinearRegression()
x = X_train[max_col].values.reshape(-1,1) # reshaping the data into the form of a list of rows
y = X_train["Rings"].values.reshape(-1,1)  # reshaping the data into the form of a list of rows
RMSE_train = []

degree = [2,3,4,5]

 # building non-linear regression models based on the degree of the polynomial
for i in degree:
    poly_features = PolynomialFeatures(i)
    x_poly = poly_features.fit_transform(x)
    regressor = LinearRegression()
    regressor.fit(x_poly,y)  # applying linear_regression for max_correl col and rings
    y_pred = regressor.predict(x_poly) # fitting the data to the model
    RMSE_train_data = mean_squared_error(y, y_pred,squared = False)
    RMSE_train.append(RMSE_train_data)
    print("The RMSE value is",RMSE_train_data,"when the value of degree is",i)


plt.plot(degree,RMSE_train,color = 'yellow')
plt.scatter(degree,RMSE_train,color = 'red',s = 100,edgecolor = "black")
plt.xlabel("Degree of polynomial")
plt.ylabel("RMSE value")
plt.title("Variation of RMSE w.r.t Degree of polynomial for train data")
plt.show()

RMSE_test = []

for i in degree:
    poly_features  =  PolynomialFeatures(i)
    x_poly = poly_features.fit_transform(x)
    regressor = LinearRegression()
    regressor.fit(x_poly,y) # applying non linear_regression for max_correl col and rings
    x_poly_test = poly_features.fit_transform(x1) # fitting the data to the model
    y_pred_test = regressor.predict(x_poly_test) # predicting the values
    RMSE_test_data = mean_squared_error(y1, y_pred_test,squared = False)
    RMSE_test.append(RMSE_test_data)
    print("The RMSE value is",RMSE_test_data,"when the value of degree is",i)


print(RMSE_test)
plt.plot(degree,RMSE_test,color = 'yellow')
plt.scatter(degree,RMSE_test,color = 'red',s = 100,edgecolor = "black")
plt.xlabel("Degree of polynomial")
plt.ylabel("RMSE value")
plt.title("Variation of RMSE w.r.t Degree of polynomial for test data")
plt.show()

 # Here RMSE value is minimum for i = 4, therefore we will use that to show the results

poly_features  =  PolynomialFeatures(4)
x_poly = poly_features.fit_transform(x)
regressor = LinearRegression()
regressor.fit(x_poly,y)
y_pred = regressor.predict(x_poly)


plt.scatter(x,y,color = "orange",label = "Training data",edgecolor = 'black')
plt.scatter(x,y_pred,color = "Brown",label = "Best fit line",marker = "*")
plt.xlabel(max_col)
plt.ylabel("Rings")
plt.legend()
plt.show()

poly_features  =  PolynomialFeatures(4)
x_poly = poly_features.fit_transform(x)
regressor = LinearRegression()
regressor.fit(x_poly,y)
x_poly_test = poly_features.fit_transform(x1)
y_pred_test = regressor.predict(x_poly_test)


plt.scatter(y1,y_pred_test,edgecolor = "red")
plt.xlabel("Actual Rings")
plt.ylabel("Calculated Rings")
plt.title("scatter plot of actual Rings vs predicted Rings on test data")
plt.show()

#Question4


RMSE_train=[]
k=[2,3,4,5]
for i in k:
    poly_features = PolynomialFeatures(i) #Applying univariate non-Linear Regression model
    X_train1=X_train.copy()
    del X_train1["Rings"]
    x_poly=poly_features.fit_transform(X_train1)
    regressor=LinearRegression()
    regressor.fit(x_poly,X_train["Rings"])

    y_pred=regressor.predict(x_poly) #Predicting the train data using constructed model
    RMSE_train_data=mean_squared_error(y, y_pred,squared=False) #Calculating the RMSE
    print("Root mean squared error on Train data in multivariate non-linear regression model for Features",i,"is:",RMSE_train_data)
    RMSE_train.append(RMSE_train_data)

#Ploting Variation of RMSE w.r.t Degree of polynomial for train data
plt.plot(k,RMSE_train,color='yellow')
plt.scatter(k,RMSE_train,color='red', edgecolor = "black",s=100)
plt.xlabel("Degree of polynomial")
plt.ylabel("RMSE value")
plt.title("Variation of RMSE w.r.t Degree of polynomial for train data")
plt.show()


RMSE_train=[]
k=[2,3,4,5]
for i in k:

    poly_features = PolynomialFeatures(i)
    X_train1=X_train.copy()
    del X_train1["Rings"]
    x_poly=poly_features.fit_transform(X_train1)
    regressor=LinearRegression()
    regressor.fit(x_poly,X_train["Rings"])
    X_test1=X_test.copy()
    del X_test1["Rings"]
    x_poly_test=poly_features.fit_transform(X_test1)
    
    y_pred_test=regressor.predict(x_poly_test) #Predicting the train data using constructed model

    RMSE_test_data=mean_squared_error(X_test["Rings"], y_pred_test,squared=False) #Calculating the RMSE
    print("Root mean squared error on Test data in multivariate non-linear regression model for Features",i,"is:",RMSE_test_data)
    RMSE_train.append(RMSE_test_data)


#Ploting Variation of RMSE w.r.t Degree of polynomial for test data
plt.plot(k,RMSE_train,color='yellow')
plt.scatter(k,RMSE_train,color='red',s=100,edgecolor = "black")
plt.xlabel("Degree of polynomial")
plt.ylabel("RMSE value")
plt.title("Variation of RMSE w.r.t Degree of polynomial for test data")
plt.show()


#Here best degree is 2
#Ploting scatter plot of actual Rings vs predicted Rings on test data
poly_features = PolynomialFeatures(2)
X_train1=X_train.copy()
del X_train1["Rings"]
x_poly=poly_features.fit_transform(X_train1)
regressor=LinearRegression()
regressor.fit(x_poly,X_train["Rings"])
X_test1=X_test.copy()
del X_test1["Rings"]
x_poly_test=poly_features.fit_transform(X_test1)


plt.scatter(X_test["Rings"],y_pred_test,color = "green",edgecolor = "black")
plt.xlabel("Actual Rings")
plt.ylabel("Calculated Rings")
plt.title("scatter plot of actual Rings vs predicted Rings on test data")
plt.show()
