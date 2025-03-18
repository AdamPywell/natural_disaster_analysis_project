# Jack Clarke
# https://www.kaggle.com/datasets/andrewmvd/tsunami-dataset
# https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def get_year(date):
    return int(date[0:4])

data = pd.read_csv("../data/tsunami.csv")
data = data[data['YEAR'] > 1900]
group = data.value_counts('YEAR')
paired = data.join(group, on='YEAR', rsuffix='_r')

# =============== Plot ===============

plt.plot(paired['YEAR'], paired['count'], 'b.', alpha=0.5)

# =============== Regression ===============

X = paired['YEAR'].values.reshape(-1, 1)
y = paired['count'].values

poly = PolynomialFeatures(degree=8);
X_poly = poly.fit_transform(X)
lr = LinearRegression(fit_intercept=False)
lr.fit(X_poly, y)

X_range = pd.DataFrame({'year_range': range(X.min(), X.max())})
plt.plot(X_range['year_range'], lr.predict(poly.transform(X_range)), color='red')

# =============== Graph ===============

plt.title('Tsunami Occurances')
plt.legend(['Count', 'Regression'])
plt.xlabel('Year')
plt.ylabel('Count')
plt.show()
# plt.savefig("tsunami.png")