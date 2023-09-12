import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
import matplotlib.pyplot as plt

crime_data = pd.read_csv('crime.csv')
crime_data.dropna(inplace=True)

label_encoder = LabelEncoder()
crime_data['TYPE'] = label_encoder.fit_transform(crime_data['TYPE'])

X = crime_data[['Latitude', 'Longitude', 'TYPE']]
y = crime_data['NEIGHBOURHOOD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training a Random Forest Classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Making predictions
y_pred = clf.predict(X_test)

# Evaluating the model
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

gdf = gpd.GeoDataFrame(crime_data, geometry=gpd.points_from_xy(crime_data.Longitude, crime_data.Latitude))

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(10, 10), color='white', edgecolor='black')
gdf.plot(ax=ax, marker='o', color='red', markersize=5, alpha=0.5)
plt.title('Crime Hotspots')
plt.show()

import streamlit as st


# Load the dataset
@st.cache  # Caching for improved performance
def load_data():
    data = pd.read_csv(
        '/Users/lucifarroy/pythonProject/DIY Dataset/crime.csv')  # file path
    return data


data = load_data()

# Create a Streamlit web app
st.title('Crime Data Filter')

# Create input widgets for filtering
crime_type = st.sidebar.selectbox('Select Crime Type:', data['TYPE'].unique())
neighborhood = st.sidebar.selectbox('Select Neighborhood:', data['NEIGHBOURHOOD'].unique())
year = st.sidebar.selectbox('Select Year:', data['YEAR'].unique())

# Filter the data based on user input
filtered_data = data[
    (data['TYPE'] == crime_type) &
    (data['NEIGHBOURHOOD'] == neighborhood) &
    (data['YEAR'] == year)
    ]

# Display the filtered data
st.write(f"Displaying {len(filtered_data)} records")
st.write(filtered_data)
