# Jack Clarke
# https://www.kaggle.com/datasets/headsortails/us-natural-disaster-declarations

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def get_year(date):
    return int(date[0:4])

disaster_data = pd.read_csv("../data/disaster_decloration.csv")
disaster_data = disaster_data[disaster_data['incident_type'] == 'Flood']
disaster_data['year'] = disaster_data['declaration_date'].apply(get_year)

# =============== Bar ===============

grp = disaster_data[['year', 'disaster_number']].groupby('year').count()
plt.bar(grp.index, grp['disaster_number'])

# =============== Prediction ===============

X = grp.index.values.reshape(-1, 1)
y = grp['disaster_number'].values

model = LinearRegression()
model.fit(X, y)
grp['predicted_count'] = model.predict(X)
plt.plot(grp.index, grp['predicted_count'], color='red', linewidth=2, label='Regression Line')

# =============== Graph ===============

plt.xlabel('Year')
plt.ylabel('Count')
plt.title("Flood Occurrences")
plt.legend(['Regression', 'Count'])

plt.show()
# plt.savefig("amount.png")