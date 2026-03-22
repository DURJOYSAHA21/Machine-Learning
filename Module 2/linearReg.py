import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.metrics import r2_score

df=pd.read_csv("FuelConsumption.csv")
print(df.head())
#pd.set_option('display.max_columns', None)
#print(df.describe())

idf=df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(idf.head())
idf.hist() #shows histogram
plt.show()

plt.scatter(idf.FUELCONSUMPTION_COMB, idf.CO2EMISSIONS)
plt.xlabel("FUEL CONSUM")
plt.ylabel("EMISSION")
plt.show()

msk=np.random.rand(len(df))<0.8
train=idf[msk]
test=idf[~msk]
print(test)

reg = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

reg.fit(train_x, train_y)

print("Coefficient",reg.coef_)
print("intercept", reg.intercept_)

plt.scatter(train.ENGINESIZE,train.CO2EMISSIONS,color='blue')
plt.plot(train_x,reg.intercept_[0] +reg.coef_[0][0]*train_x,'-r')
plt.show()

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

test_y_ =reg.predict(test_x)
print("MEAN ABS ERROR", np.mean(np.absolute(test_y_ -test_y)))
print("MEAN SQUARED ERROR", np.mean((test_y_ -test_y)**2))
print("R2:", r2_score(test_y,test_y_))