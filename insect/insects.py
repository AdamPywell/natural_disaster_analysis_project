#!/usr/bin/env python
# coding: utf-8

# In[ ]:
# Morgan Hindy
# https://www.kaggle.com/datasets/brsdincer/all-natural-disasters-19002021-eosdis
# https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr
import sys

def model_data(merged_data_all, deg):
    X_insect_all = merged_data_all['Decade'].values.reshape(-1, 1)
    y_insect_all = merged_data_all['Insect Count'].values
    
    poly_insect_all = PolynomialFeatures(degree=deg)
    X_insect_poly_all = poly_insect_all.fit_transform(X_insect_all)
    
    model_insect_all = LinearRegression()
    model_insect_all.fit(X_insect_poly_all, y_insect_all)
    merged_data_all['Insect Count Prediction'] = model_insect_all.predict(X_insect_poly_all)
    
    X_temp_all = merged_data_all['Year'].values.reshape(-1, 1)
    y_temp_all = merged_data_all['LandAndOceanAverageTemperature'].values
    
    poly_temp_all = PolynomialFeatures(degree=8)
    X_temp_poly_all = poly_temp_all.fit_transform(X_temp_all)
    
    model_temp_all = LinearRegression(fit_intercept=False)
    model_temp_all.fit(X_temp_poly_all, y_temp_all)
    merged_data_all['Temp Prediction'] = model_temp_all.predict(X_temp_poly_all)
    
    # Scale temperature and epidemic count values for visual comparison
    scaler = StandardScaler()
    merged_data_all[['Insect Prediction Scaled', 'Temp Prediction Scaled']] = scaler.fit_transform(
        merged_data_all[['Insect Count Prediction', 'Temp Prediction']]
    )
    
    # Plot both regressions together for all data
    plt.figure(figsize=(10, 5))
    plt.plot(merged_data_all['Decade'], merged_data_all['Temp Prediction Scaled'], color='red', label='Temp')
    plt.plot(merged_data_all['Decade'], merged_data_all['Insect Prediction Scaled'], color='blue', label='Insect Infestations')
    plt.legend(['Land/Ocean Avg Temp', 'Insect Infestations'])
    plt.xlabel('Decade')
    plt.ylabel('Value')
    plt.title("Avg Global Temperature vs Insect Infestation Prediction")
    plt.savefig('insects_decade_temp.png')
    
    # Plot Insect Counts vs year for all data
    plt.figure(figsize=(10, 5))
    plt.plot(merged_data_all['Decade'], merged_data_all['Insect Count Prediction'], color='red', label='Regression')
    plt.plot(merged_data_all['Decade'], merged_data_all['Insect Count'], 'b.', label='Insect Infestations')
    plt.legend(['Regression', 'Insect Infestations'])
    plt.xlabel('Year')
    plt.ylabel('Count')
    plt.title("Insect Infestation Count per Decade vs Regression")
    plt.savefig('insects_decade_regression.png')
    
    # Statistical test for correlation for all data
    corr_coef_all, p_value_all = pearsonr(merged_data_all['Insect Count Prediction'], merged_data_all['Temp Prediction'])
    print(f'All Data: Correlation Coefficient: {corr_coef_all}, P-Value: {p_value_all}')


data = pd.read_csv("../data/disasters.csv", header=0, parse_dates=['Year'])
data_cols = data[['Year', 'Country', 'Disaster Type']]
insects = data_cols[data_cols['Disaster Type'] == 'Insect infestation']

# Remove duplicates and missing values
insects = insects.drop_duplicates()
insects = insects.dropna()

# Convert Year to integers
insects['Year'] = insects['Year'].dt.year.astype(int)
insects = insects[insects['Year'] <= 2024]

# Create a decade column
insects['Decade'] = (insects['Year'] // 10) * 10
# Count the number of insect infestations per decade
insect_counts_decades = insects.groupby('Decade').size().reset_index(name='Insect Count')
insect_counts_decades = insect_counts_decades[insect_counts_decades['Insect Count'] < 30]


insect_counts = insects.groupby('Year').agg({'Disaster Type' : 'count'}).rename(columns={'Disaster Type' : 'Insect Count'}).reset_index()
plt.figure(figsize=(10, 5))
plt.bar(insect_counts_decades['Decade'],insect_counts_decades['Insect Count'], width=8, label ='Insect Infestations')
plt.xticks(np.arange(insect_counts_decades['Decade'].min(), insect_counts_decades['Decade'].max() + 10, 10))  # Align bins with decades
plt.xlabel('Decade')
plt.ylabel('Insect Infestation Count')
plt.title("Insect Infestation Count per Decade")
plt.savefig('insects_decades.png')

# # Remove outliers using IQR
# print(insect_counts['Insect Count'].max(), insect_counts['Insect Count'].min())
# Q1 = insect_counts['Insect Count'].quantile(0.25)
# Q3 = insect_counts['Insect Count'].quantile(0.75)
# IQR = Q3 - Q1
# insects_cleaned = insect_counts[~((insect_counts['Insect Count'] < (Q1 - 1.5 * IQR)) | (insect_counts['Insect Count'] > (Q3 + 1.5 * IQR)))]

# Load and clean temperature data
def get_year(date):
    return int(date[0:4])

temp_data = pd.read_csv("../data/global_temp.csv")
temp_data['Year'] = temp_data['dt'].apply(get_year)
temp_data = temp_data[temp_data['Year'] > 1901]
grp_temp = temp_data[['Year', 'LandAndOceanAverageTemperature']].groupby('Year').agg('mean').dropna().reset_index()

# Analysis with decades
merged_data_decades = pd.merge(insect_counts_decades, grp_temp, left_on='Decade', right_on='Year')
model_data(merged_data_decades, 3)

