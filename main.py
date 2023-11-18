import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets,linear_model
from sklearn.metrics import mean_squared_error
import pandas as pd

datasetss = datasets.load_diabetes()
# pds=datasetss.data
# print(len(pds))
print(datasetss.keys())
dataset = datasetss.data[:,np.newaxis,2]
dataset_x_train = dataset[:-221]
dataset_x_test = dataset[-221:]
#
dataset_y_train = datasetss.target[:-221]
dataset_y_test = datasetss.target[-221:]
#
reg = linear_model.LinearRegression()
reg.fit(dataset_x_train,dataset_y_train)
#
y_predict = reg.predict(dataset_x_test)
#
accuracy = mean_squared_error(dataset_y_test,y_predict)
#
print(accuracy)
# weights = reg.coef_
# intercept = reg.intercept_
# # print(weights,intercept)
#
plt.scatter(dataset_x_test,dataset_y_test)
plt.plot(dataset_x_test,y_predict)
plt.show()






