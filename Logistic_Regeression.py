import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import metrics

# Column names for the data
colnames = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', "class"]
pima_df= pd.read_csv("D:\PyCharm Community Edition 2019.1.3\Logistic_Regression\pima-indians-diabetes-2.data",names = colnames)
print(pima_df.head())

# Checking for srings in dataset
print(pima_df[~pima_df.applymap(np.isreal).all(1)])

#pima_df = pima_df.fillna(pima_df.median())

# Looking for distribution of classes
groups=pima_df.groupby(["class"]).count()
print(groups)

sns.pairplot(pima_df , diag_kind= 'kde' ,hue = 'class')
# plt.show()

#Separating training and testing set
X = pima_df.drop('class', axis=1)
y = pima_df[['class']]
seed = 7
X_train,X_test,y_train,y_test = train_test_split( X, y ,test_size= 0.3,random_state = seed )

model = LogisticRegression()
model.fit(X_train,y_train)
y_predict = model.predict(X_test)
model_score = model.score(X_test, y_test)
print(model_score)
print(metrics.confusion_matrix(y_test,y_predict))

from sklearn.svm import SVC
plot_confusion_matrix(model, X_test, y_test)
plt.show()

from sklearn import preprocessing
X_train_scaled = preprocessing.scale(X_train)
X_test_scaled = preprocessing.scale(X_test)
model2 = LogisticRegression()
model2.fit(X_train_scaled, y_train)
y_predict2 = model2.predict(X_test_scaled)
model2_score = model2.score(X_test_scaled, y_test)
print(model_score)
plot_confusion_matrix(model2, X_test_scaled, y_test)
plt.show()