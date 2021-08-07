import pandas as pd
import numpy as np
import quandl, math, datetime
from sklearn import preprocessing, svm, cross_validation
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import pickle

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

#Preprocessing or normalizing data/features
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X = X[:-forecast_out]
X_later = X[-forecast_out:]

df.dropna(inplace = True)
y = np.array(df['label'])

#splitting into train and test set
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size = 0.2)

###decide the classifier and fit it onto the data
##clf = LinearRegression()
##clf.fit(X_train, y_train)
##
###saving the classifer
##with open('linearregression.pickle', 'wb') as f:
##    pickle.dump(clf, f)

#load the classifier
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in) 

#get the accuracy
accuracy = clf.score(X_test, y_test)

#print(len(X), len(y))
print(accuracy)

#forecasting out predicted values
forecast_set = clf.predict(X_later)

print(forecast_set, accuracy, forecast_out)

#testing CI
#visualizing data using graph
last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
oneday = 86400
next_unix = last_unix + oneday

df['Forecast'] = np.nan

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix+=86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
    

