import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions

df =pd.read_csv("G:\Files which are needed\Python\Machine Learning\placement.csv")
print(df.head())

print(df.shape)

#preprocess--> remove outliers missing values
#EDA
#feature selection
#extract input and output cells
#scale the values
#train test split
#train the model
#evaluate model
#deploy

#preprocess
df.info()
df=df.iloc[:,1:]
print(df.head())

plt.scatter(df['cgpa'],df['iq'],c=df['placement'])
plt.show()

x=df.iloc[:,0:2]
y=df.iloc[:,-1]


xtrain,xtest,ytrain,ytest=(train_test_split(x,y,test_size=0.1))

#scale
scale=StandardScaler() #jate sob 0-1 range a ashe
xtrain=scale.fit_transform(xtrain)
xtest=scale.transform(xtest)

clf=LogisticRegression()

clf.fit(xtrain,ytrain)
ypre=clf.predict(xtest)
print(ypre)
print(ytest)

print(accuracy_score(ytest,ypre))
plot_decision_regions(xtrain, ytrain.values, clf=clf, legend=2) #to visualize how the decision line devide the pattern
plt.show()

