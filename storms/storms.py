#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# Morgan Hindy
# https://www.ncei.noaa.gov/products/international-best-track-archive

import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

# Load and clean storm data
storm_file = sys.argv[1]
storm_data = pd.read_csv(storm_file, low_memory=False).dropna().reset_index(drop=True)
storm_data = storm_data[1:]
storm_data = storm_data.astype({'SEASON': 'int32'})

# Group and aggregate storm data
count_data = storm_data.groupby('SEASON').agg({'SEASON': 'count'})
storm_data = storm_data.join(count_data, on='SEASON', rsuffix='_')
storm_data = storm_data.rename(columns={'SEASON_': 'Storm Counts'})
storm_data = storm_data[storm_data['SEASON'] > 1899]
storm_data = storm_data.drop_duplicates(subset='SEASON').reset_index(drop=True)

# Load and clean temperature data
temp_file = sys.argv[2]
def get_year(date):
    return int(date[0:4])

temp_data = pd.read_csv(temp_file)
temp_data['year'] = temp_data['dt'].apply(get_year)
temp_data = temp_data[temp_data['year'] > 1901]
grp_temp = temp_data[['year', 'LandAndOceanAverageTemperature']].groupby('year').agg('mean').dropna().reset_index()

# Rename 'year' column to 'SEASON' to match storm data
grp_temp = grp_temp.rename(columns={'year': 'SEASON'})

# Merge storm data with temperature data
merged_data = pd.merge(storm_data, grp_temp, on='SEASON')

# Prepare the data for polynomial regression on storm counts
X_storm = merged_data['SEASON'].values.reshape(-1, 1)
y_storm = merged_data['Storm Counts'].values

# Transform storm features to polynomial features
poly_storm = PolynomialFeatures(degree=3)
X_storm_poly = poly_storm.fit_transform(X_storm)

# Fit the polynomial regression model for storm data
model_storm = LinearRegression()
model_storm.fit(X_storm_poly, y_storm)

# Make predictions for storm data
merged_data['Storm Count Prediction'] = model_storm.predict(X_storm_poly)

# Prepare temperature data for polynomial regression
X_temp = merged_data['SEASON'].values.reshape(-1, 1)
y_temp = merged_data['LandAndOceanAverageTemperature'].values

# Transform temperature features to polynomial features
poly_temp = PolynomialFeatures(degree=8)
X_temp_poly = poly_temp.fit_transform(X_temp)

# Fit the polynomial regression model for temperature data
model_temp = LinearRegression(fit_intercept=False)
model_temp.fit(X_temp_poly, y_temp)

# Make predictions for temperature data
merged_data['Temp Prediction'] = model_temp.predict(X_temp_poly)

# Scale temperature and storm count values for visual comparison
scaler = StandardScaler()
merged_data[['Storm Prediction Scaled', 'Temp Prediction Scaled']] = scaler.fit_transform(merged_data[['Storm Count Prediction', 'Temp Prediction']])

# Plot both regressions together
plt.figure(figsize=(18, 5))
plt.plot(merged_data['SEASON'], merged_data['Temp Prediction Scaled'], color='red', label='Temp')
plt.plot(merged_data['SEASON'], merged_data['Storm Prediction Scaled'], color='blue', label='Storms')

plt.legend(['Land/Ocean Avg Temp', 'Storms'])
plt.xlabel('Year')
plt.ylabel('Value')
plt.title("Avg Global Temperature vs Storm Count Prediction")

plt.show()

# Plot Storm Counts vs year
plt.figure(figsize=(18, 5))
plt.plot(merged_data['SEASON'], merged_data['Storm Count Prediction'], color='red', label='Regression')
plt.plot(merged_data['SEASON'], merged_data['Storm Counts'], 'b.', label='Storms')

plt.legend(['Regression', 'Storms'])
plt.xlabel('Year')
plt.ylabel('Count')
plt.title("Yearly Storm Count vs Regression")

plt.show()
# Statistical test for correlation
corr_coef, p_value = pearsonr(merged_data['Storm Count Prediction'], merged_data['Temp Prediction'])

print(f'Correlation Coefficient: {corr_coef}')
print(f'P-Value: {p_value}')



# In[ ]:




