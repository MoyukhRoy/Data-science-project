#Run the program in googel colab or Virtual environment
#but u must install the below libery
#pip install pandas-profiling

import pandas as pd
from pandas_profiling import ProfileReport
df = pd.read_csv('/content/Country-data.csv')
print(df.head())
prof=ProfileReport(df)
prof.to_file(output_file = 'output.html')
