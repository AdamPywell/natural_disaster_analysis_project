# Jack Clarke
# https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def get_year(date):
    return int(date[0:4])


data = pd.read_csv("../data/global_temp.csv")
data['year'] = data['dt'].apply(get_year)
data = data[data['year'] > 1901]

# =============== Actual ===============

grp = data[['year', 'LandAndOceanAverageTemperature']].groupby('year').agg('mean').dropna()
plt.plot(grp.index, grp['LandAndOceanAverageTemperature'], 'b.', alpha=0.5)

# =============== Prediction ===============

X = grp.index.values.reshape(-1, 1)
y = grp['LandAndOceanAverageTemperature'].values

poly = PolynomialFeatures(degree=8);
X_poly = poly.fit_transform(X)
lr = LinearRegression(fit_intercept=False)
lr.fit(X_poly, y)

X_range = pd.DataFrame({'year': range(X.min(), X.max())})
plt.plot(X_range['year'], lr.predict(poly.transform(X_range)), color='red', label='Polynomial Fit')

# =============== Graph ===============

plt.legend(['Land/Ocean Avg Temp', 'Regression'])
plt.xlabel('Year')
plt.ylabel('Temp (c)')
plt.title("Avg Global Temperature")

plt.show()
# plt.savefig("temp.png")