# Jack Clarke
# https://www.kaggle.com/datasets/andrewmvd/tsunami-dataset
# https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data

import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def get_year(date):
    return int(date[0:4])

# =============== Plot 1: tsunami ===============

data = pd.read_csv("../data/tsunami.csv")
data = data[data['YEAR'] > 1900]
group = data.value_counts('YEAR')
paired = data.join(group, on='YEAR', rsuffix='_r')

# Ref https://matplotlib.org/stable/gallery/subplots_axes_and_figures/two_scales.html
fig, ax1 = plt.subplots()

ax1.set_xlabel('Year')
ax1.set_ylabel('Tsunami Occurances')
ax1.tick_params(axis='y', labelcolor='red')

X = paired['YEAR'].values.reshape(-1, 1)
y = paired['count'].values

poly = PolynomialFeatures(degree=8);
X_poly = poly.fit_transform(X)
lr = LinearRegression(fit_intercept=False)
lr.fit(X_poly, y)

X_range = pd.DataFrame({'year_range': range(X.min(), X.max())})

ax1.plot(X_range['year_range'], lr.predict( poly.transform(X_range)), color='red')

# =============== Plot 2: global temp ===============

data = pd.read_csv("../data/global_temp.csv")
data['year'] = data['dt'].apply(get_year)
data = data[data['year'] > 1900]
grp = data[['year', 'LandAndOceanAverageTemperature']].groupby('year').agg('mean').dropna()

X = grp.index.values.reshape(-1, 1)
y = grp['LandAndOceanAverageTemperature'].values

poly = PolynomialFeatures(degree=8);
X_poly = poly.fit_transform(X)
lr = LinearRegression(fit_intercept=False)
lr.fit(X_poly, y)

X_range = pd.DataFrame({'year_range': range(X.min(), X.max())})

ax2 = ax1.twinx()
ax2.set_ylabel('Avg Temp (c)')
ax2.plot(X_range['year_range'], lr.predict(poly.transform(X_range)), color='blue')
ax2.tick_params(axis='y', labelcolor='blue')

# =============== Graph ===============

plt.title("Avg Temp vs Tsunami Occurrences")
plt.show()
# plt.savefig("tsunami.png")

# ==== Correlation ====

corr = stats.pearsonr(grp['LandAndOceanAverageTemperature'], grp.index)
print("CORRELATION: ")
print(corr.statistic)
print(corr.pvalue)