import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor

df = pd.read_csv("DIY Dataset/insurance.csv")
print(df, df.isna().sum())
sns.boxplot(df['charges'])
plt.show()
hp = sorted(df['charges'])
q1, q3 = np.percentile(hp, [25, 75])
lower_bound = q1 - (1.5 * (q3 - q1))
upper_bound = q3 + (1.5 * (q3 - q1))
below = df['charges'] > lower_bound
above = df['charges'] < upper_bound
new_df = df[below & above]
fullRaw2 = pd.get_dummies(new_df).copy()
print(fullRaw2.shape, fullRaw2.head())
x = fullRaw2.drop(["charges"], axis=1).copy()
y = fullRaw2["charges"].copy()
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=100)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
model = LinearRegression().fit(x_train, y_train)
score1 = model.score(x_test, y_test)
1 - (1 - model.score(x_test, y_test)) * (len(y_test) - 1) / (len(y_test) - x_test.shape[1] - 1)
fig, axes = plt.subplots(1, 1)
fig.suptitle('[Residual Plots]')
fig.set_size_inches(12, 5)
axes.plot(model.predict(x_test), y_test - model.predict(x_test), 'bo')
axes.axhline(y=0, color='k')
axes.grid()
axes.set_title('Linear')
axes.set_xlabel('predicted values')
axes.set_ylabel('residuals')
plt.show()
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(x.values, i) for i in range(x.shape[1])]
vif["features"] = x.columns
print(vif["VIF"])
ridgeReg = Ridge(alpha=0.00001)
x3 = fullRaw2.drop(["charges"], axis=1).copy()
y3 = fullRaw2["charges"].copy()
x_train3, x_test3, y_train3, y_test3 = train_test_split(x3, y3, test_size=0.20, random_state=150)

ridgeReg.fit(x_train3, y_train3)
pred = ridgeReg.predict(x_test3)
score4 = ridgeReg.score(x_test3, y_test3)
lassoReg = Lasso(alpha=0.0001)
lassoReg.fit(x_train3, y_train3)
pred = lassoReg.predict(x_test3)
score5 = lassoReg.score(x_test3, y_test3)

print(score4)
print(score5)
print("all model score is:")
print("simple linear regression:          ", score1)
print("ridge regression:                  ", score4)
print("lasso regression:                  ", score5)
