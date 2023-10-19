import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

from scipy import stats
stats.chisqprob = lambda chisq, df: stats.chi2.sf(chisq, df)

raw_data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/LOGISTIK REG/2.02.Binary predictors.csv')
data = raw_data.copy()

data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})
data['Gender'] = data['Gender'].map({'Female':1,'Male':0})
#print(data)

y = data['Admitted']
x1 = data[['SAT','Gender']]


x = sm.add_constant(x1)
reg_log = sm.Logit(y, x)
results_log = reg_log.fit()
print(results_log.summary())
print(np.exp(1.9449))

def f(x,b0,b1):
    return np.array(np.exp(b0+x*b1)/(1+ np.exp(b0+x*b1)))

f_sorted = np.sort(f(x1['Gender'],results_log.params[0],results_log.params[1]))
x_sorted = np.sort(np.array(x1['Gender']))

plt.scatter(data['Gender'],y, color='red')
plt.xlabel('Gender')
plt.ylabel('Admitted')
plt.plot(x_sorted,f_sorted, color='blue')
plt.show()

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
#print(results_log.predict())
np.array(data['Admitted'])
print(results_log.pred_table())

#CM (CUNFUSION MATRIX)
cm_df = pd.DataFrame(results_log.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0', 1:'Actual 1'})
print(cm_df)

#AKURASI MODEL TRAIN
cm = np.array(cm_df)
accurancy_train = (cm[0,0]+cm[1,1])/cm.sum()
print(accurancy_train)

#TESTING
test1 = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/LOGISTIK REG/2.03.Test dataset.csv')
test = test1.copy()

test['Admitted'] = test['Admitted'].map({'Yes':1, 'No':0})
test['Gender'] = test['Gender'].map({'Female':1,'Male':0})

test_y = test['Admitted']
test_x = test.drop(['Admitted'], axis=1)
test_data = sm.add_constant(test_x)

def confusion_matrix(data,actual_values,model):
    
        pred_values = model.predict(data)
        bins=np.array([0,0.5,1])
        cm = np.histogram2d(actual_values, pred_values, bins=bins)[0]
        accuracy = (cm[0,0]+cm[1,1])/cm.sum()
        return cm, accuracy

cm = confusion_matrix(test_data,test_y,results_log)
print(cm)

cm_df = pd.DataFrame(cm[0])
cm_df.columns = ['Predicted 0','Predicted 1']
cm_df = cm_df.rename(index={0: 'Actual 0',1:'Actual 1'})
print(cm_df)

print ('Missclassification rate: '+str((1+1)/19))