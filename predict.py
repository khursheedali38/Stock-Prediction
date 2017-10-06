import pandas as pd
import numpy as np
import quandl, math
from sklearn import preprocessing, svm, cross_validation
from sklearn.linear_model import LinearRegression

#get data
df = quandl.get('WIKI/GOOGL')
df = df[['Adj. Close', 'Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Volume']]

#get the required data for training
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low'])/df['Adj. Close'] * 100.0
df['PCT'] = (df['Adj. Close'] - df['Adj. Open'])/df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT', 'Adj. Volume']]

forecast_col = 'Adj. Close'

#handle missing data
df.fillna(-9999, inplace = True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace = True)

#Preprocessing or normalizing data/features
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
y = np.array(df['label'])

#splitting into train and test set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

#decide the classifier and fit it onto the data
clf = svm.SVR()
clf.fit(X_train, y_train)

#get the accuracy
accuracy = clf.score(X_test, y_test)

#print(len(X), len(y))
print(accuracy)
