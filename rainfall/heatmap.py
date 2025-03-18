# Jack Clarke
# https://www.kaggle.com/datasets/zusmani/rainfall-in-pakistan
# https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt

def get_year(date):
    return int(date[0:4])

def to_ordinal(row):
    return dt.datetime(int(row['Year']), int(row['Month']), 1).toordinal()

# =============== Rainfall ===============

data = pd.read_csv("../data/rainfall.csv")
data.columns = data.columns.str.strip()
data['Rainfall'] = data['Rainfall - (MM)']

DATE_TO_INT_MAP = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12
}

data['Month'] = data['Month'].map(DATE_TO_INT_MAP)
data['Ordinal'] = data.apply(to_ordinal, axis=1)

grp = data.groupby('Year').agg({'Rainfall - (MM)': 'mean'})
data = data.join(grp, on='Year', rsuffix='_r')

# =============== Global temps ===============

temp_data = pd.read_csv("../data/global_temp.csv")
temp_data['year'] = temp_data['dt'].apply(get_year)
temp_data = temp_data[temp_data['year'] >= 1901] # To align with rainfall data
temp_data = temp_data[temp_data['year'] <= 2016]  # To align with rainfall data

data = data.drop(data.tail(12).index) # To align dates by a year
data['Land Temp'] = temp_data['LandAndOceanAverageTemperature'].values
data['Land+Ocean Temp'] = temp_data['LandAverageTemperature'].values

corr = data[['Rainfall', 'Ordinal', 'Month', 'Land Temp', 'Land+Ocean Temp']].corr()
sns.heatmap(corr, annot=True, cmap='bwr', fmt='.2f')
plt.title("Rainfall Heatmap")
plt.tight_layout(pad=0.5)

plt.show()
# plt.savefig("heatmap.png")
