import numpy as np
import pandas as pd
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/LOGISTIK REG/Bank data.csv')
data = raw_data.copy()
data = data.drop(['Unnamed: 0'], axis=1)

data['y'] = data['y'].map({'yes': 1, 'no': 0})
#print(data.describe())

y = data['y']
x1 = data['duration']

x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()
print(results_log.summary())

def f(x,b0,b1):
    return np.array(np.exp(b0+x*b1) / (1 + np.exp(b0+x*b1)))

f_sorted = np.sort(f(x1,results_log.params[0],results_log.params[1]))
x_sorted = np.sort(np.array(x1))

plt.scatter(x1,y, color='red')
plt.xlabel('Duration')
plt.ylabel('Subscription')
plt.plot(x_sorted,f_sorted,color='C8')
plt.show()


plt.scatter(x1, y, color='red')
plt.xlabel('Duration')
plt.ylabel('Subcription')
plt.show()

#ODDS OF DURATION
print(np.exp(0.0051))

#REGRESI MULTIVARIATE
estimators = ['interest_rate','march','credit','previous','duration']

y = data['y']
X1 = data[estimators]
print(X1.describe())

X = sm.add_constant(X1)
reg_logit = sm.Logit(y,X)
results_logit = reg_logit.fit()
print(results_logit.summary())

#AKURASI TRAIN DATA
def confusion_matrix(data,actual_values,model):
    pred_values = model.predict(data)
    bins = np.array([0,0.5,1])
    cm = np.histogram2d(actual_values,pred_values,bins=bins)[0]
    accurancy = (cm[0,0]+cm[1,1])/cm.sum()
    return cm, accurancy

print(confusion_matrix(X, y, results_logit))

#TESTING MULTIVARIATE
raw_data2 = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/LOGISTIK REG/Bank data testing.csv')
data_test = raw_data2.copy()
data_test = data_test.drop(['Unnamed: 0'], axis=1)
data_test['y'] = data_test['y'].map({'yes':1, 'no':0})
print(data_test)

y_test = data_test['y']
X1_test = data_test[estimators]
X_test = sm.add_constant(X1_test)

print(confusion_matrix(X_test, y_test, results_logit))