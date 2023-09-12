import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/lucifarroy/pythonProject/DIY Dataset/housing_price.csv")
df.isna().sum()
x = df.drop(columns="median_house_value").values
y = df.median_house_value.values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=10)
print(df, df.isna().sum(), ("independent data\n", x), ("\ndependent data\n", y),
      ("x_train and x_test dataset shape", x_train.shape, x_test.shape),
      ("y_train and y_test dataset shape", y_train.shape, y_test.shape))
regressor = LinearRegression()
regressor.fit(x_train, y_train)
regressor.score(x_test, y_test)
print(("R2 value:", regressor.score(x_test, y_test)),
      ("\ncoefficient: \n ", regressor.coef_), ("\nintercept:", regressor.intercept_))
y_pared = regressor.predict(x_test)
pared = pd.DataFrame({'Actual': y_test, 'Predicted': y_pared})
df1 = pared.head(10)
print(df1)
X2 = sm.add_constant(x)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset
# Assuming your dataset is stored in a CSV file named 'housing_data.csv'
data = pd.read_csv('/Users/lucifarroy/pythonProject/DIY Dataset/housing_price.csv')

# Streamlit app title and introduction
st.title('Housing Data Visualization App')
st.write('Explore and visualize the housing dataset.')

# Display the dataset as a table
st.write('## Housing Data Overview')
st.dataframe(data)

# Sidebar filters
st.sidebar.header('Data Filters')

# Filtering options
min_age = st.sidebar.slider('Minimum Housing Age', int(data['housing_median_age'].min()), int(data['housing_median_age'].max()), int(data['housing_median_age'].min()))
max_age = st.sidebar.slider('Maximum Housing Age', min_age, int(data['housing_median_age'].max()), int(data['housing_median_age'].max()))
min_income = st.sidebar.slider('Minimum Median Income', float(data['median_income'].min()), float(data['median_income'].max()), float(data['median_income'].min()))
max_income = st.sidebar.slider('Maximum Median Income', min_income, float(data['median_income'].max()), float(data['median_income'].max()))

# Apply filters
filtered_data = data[(data['housing_median_age'] >= min_age) & (data['housing_median_age'] <= max_age) & (data['median_income'] >= min_income) & (data['median_income'] <= max_income)]

# Display filtered data as a table
st.write('## Filtered Data')
st.dataframe(filtered_data)

# Data visualization
st.write('## Data Visualization')

# Scatter plot of median income vs. median house value using Matplotlib
fig, ax = plt.subplots()
ax.scatter(filtered_data['median_income'], filtered_data['median_house_value'])
ax.set_xlabel('Median Income')
ax.set_ylabel('Median House Value')
ax.set_title('Scatter Plot of Median Income vs. Median House Value')
st.pyplot(fig)

# Heatmap using Seaborn
st.write('### Heatmap')
corr = filtered_data.corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
st.pyplot(fig)

