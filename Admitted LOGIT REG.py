import numpy as np
import pandas as pd
import statsmodels.api as sm 
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

raw_data = pd.read_csv('D:/DATA ANALYST/belajar_python/LATIHAN DS/MECHINE LEARNING/LOGISTIK REG/2.01.Admittance.csv')
data = raw_data.copy()
data['Admitted'] = data['Admitted'].map({'Yes':1, 'No':0})
#print(data)

y = data['Admitted']
x1 = data['SAT']

plt.scatter(x1, y, color='red')
plt.xlabel('SAT')
plt.ylabel('Admitted')
#plt.show()

x = sm.add_constant(x1)

#PAKAI OLS
#reg_lin = sm.OLS(y,x)
#results_linreg = reg_lin.fit()

#plt.scatter(x1,y,color = 'C0')
#y_hat = x1*results_linreg.params[1] + results_linreg.params[0]
#plt.plot(x1,y_hat,lw=2.5,color='C8')
#plt.xlabel('SAT', fontsize = 20)
#plt.ylabel('Admitted', fontsize = 20)
#plt.show()

#PAKAI LOGISTIK REGRESI
reg_log = sm.Logit(y, x)
results_logreg = reg_log.fit()
print(results_logreg.summary())

#MEMBUAT KURVA LOGISTIK 
# Creating a logit function, depending on the input and coefficients
def f(x,b0,b1):
    return np.array(np.exp(b0+x*b1) / (1 + np.exp(b0+x*b1)))

# Sorting the y and x, so we can plot the curve
f_sorted = np.sort(f(x1,results_logreg.params[0],results_logreg.params[1]))
x_sorted = np.sort(np.array(x1))

plt.scatter(x1,y,color='C0')
plt.xlabel('SAT', fontsize = 20)
plt.ylabel('Admitted', fontsize = 20)
# Plotting the curve
plt.plot(x_sorted,f_sorted,color='C8')
plt.show()

np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})
np.array(data['Admitted'])
cm_df = pd.DataFrame(results_logreg.pred_table())
cm_df.columns = ['Predicted 0', 'Predicted 1']
cm_df = cm_df.rename(index={0:'Actual 0', 1:'Actual 1'})
print(cm_df)

cm = np.array(cm_df)
accurancy_train = (cm[0,0]+cm[1,1])/cm.sum()
print(accurancy_train)
