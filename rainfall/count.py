# Jack Clarke
# https://www.kaggle.com/datasets/zusmani/rainfall-in-pakistan

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("../data/rainfall.csv")
grp = data.groupby('Month').sum().reset_index()

# Converts the month to categorical so it can be sorted. Ref https://stackoverflow.com/questions/48042915/sort-a-pandas-dataframe-series-by-month-name
months = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
grp['Month'] = pd.Categorical(grp['Month'], categories=months, ordered=True)
grp = grp.sort_values('Month')

plt.xticks(rotation=45)
plt.title("Rainfall Per Month")

plt.bar(grp['Month'], grp['Rainfall - (MM)'])
plt.xlabel('Year')
plt.ylabel('Amount (MM)')
plt.tight_layout(pad=0.5)

plt.show()
# plt.savefig("heatmap.png")

