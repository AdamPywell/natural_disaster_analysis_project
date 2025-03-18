#!/usr/bin/env python
# coding: utf-8

# In[1]:
# Morgan Hindy

# https://www.kaggle.com/datasets/brsdincer/all-natural-disasters-19002021-eosdis
# https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import seaborn as sns
import matplotlib.ticker as ticker
import sys

# Load epidemic data
data = pd.read_csv("../data/disasters.csv", header=0, parse_dates=['Year'])
data_cols = data[['Year', 'Country', 'Disaster Type']]
epidemics = data_cols[data_cols['Disaster Type'] == 'Epidemic']

# Convert Year to integers
epidemics['Year'] = epidemics['Year'].dt.year.astype(int)
epidemics = epidemics[epidemics['Year'] <= 2024]

# Remove duplicates and missing values
epidemics = epidemics.drop_duplicates()
epidemics = epidemics.dropna()

# Count the number of epidemics per year
epidemics_count = epidemics.groupby('Year').size().reset_index(name='Epidemic Count')

# Log transform the counts
epidemics_count['Log Count'] = np.log1p(epidemics_count['Epidemic Count'])

# Load and clean temperature data
def get_year(date):
    return int(date[0:4])

temp_data = pd.read_csv("../data/global_temp.csv")
temp_data['Year'] = temp_data['dt'].apply(get_year)
temp_data = temp_data[temp_data['Year'] > 1901]
grp_temp = temp_data[['Year', 'LandAndOceanAverageTemperature']].groupby('Year').agg('mean').dropna().reset_index()

# Analysis with all data
merged_data_all = pd.merge(epidemics_count, grp_temp, on='Year')
X_epidemic_all = merged_data_all['Year'].values.reshape(-1, 1)
y_epidemic_all = merged_data_all['Log Count'].values

poly_epidemic_all = PolynomialFeatures(degree=3)
X_epidemic_poly_all = poly_epidemic_all.fit_transform(X_epidemic_all)

model_epidemic_all = LinearRegression()
model_epidemic_all.fit(X_epidemic_poly_all, y_epidemic_all)
merged_data_all['Epidemic Count Prediction'] = model_epidemic_all.predict(X_epidemic_poly_all)

X_temp_all = merged_data_all['Year'].values.reshape(-1, 1)
y_temp_all = merged_data_all['LandAndOceanAverageTemperature'].values

poly_temp_all = PolynomialFeatures(degree=8)
X_temp_poly_all = poly_temp_all.fit_transform(X_temp_all)

model_temp_all = LinearRegression(fit_intercept=False)
model_temp_all.fit(X_temp_poly_all, y_temp_all)
merged_data_all['Temp Prediction'] = model_temp_all.predict(X_temp_poly_all)

# Scale temperature and epidemic count values for visual comparison
scaler = StandardScaler()
merged_data_all[['Epidemic Prediction Scaled', 'Temp Prediction Scaled']] = scaler.fit_transform(
    merged_data_all[['Epidemic Count Prediction', 'Temp Prediction']]
)

# Plot both regressions together for all data
plt.figure(figsize=(10, 5))
plt.plot(merged_data_all['Year'], merged_data_all['Temp Prediction Scaled'], color='red', label='Temp')
plt.plot(merged_data_all['Year'], merged_data_all['Epidemic Prediction Scaled'], color='blue', label='Epidemics')
plt.legend(['Land/Ocean Avg Temp', 'Epidemics'])
plt.xlabel('Year')
plt.ylabel('Value')
plt.title("Avg Global Temperature vs Epidemic Count Prediction (All Data)")
plt.show()

# Plot Epidemic Counts vs year for all data
plt.figure(figsize=(10, 5))
plt.plot(merged_data_all['Year'], merged_data_all['Epidemic Count Prediction'], color='red', label='Regression')
plt.plot(merged_data_all['Year'], merged_data_all['Log Count'], 'b.', label='Epidemics')
plt.legend(['Regression', 'Epidemics'])
plt.xlabel('Year')
plt.ylabel('Count')
plt.title("Yearly Epidemic Count vs Regression (All Data)")
plt.show()

# Statistical test for correlation for all data
corr_coef_all, p_value_all = pearsonr(merged_data_all['Epidemic Count Prediction'], merged_data_all['Temp Prediction'])
print(f'All Data: Correlation Coefficient: {corr_coef_all}, P-Value: {p_value_all}')







