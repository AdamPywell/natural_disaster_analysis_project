# Jack Clarke
# https://www.kaggle.com/datasets/headsortails/us-natural-disaster-declarations
# https://www.kaggle.com/datasets/berkeleyearth/climate-change-earth-surface-temperature-data

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize

def get_year(date):
    return int(date[0:4])

def get_date(date):
    return str(date[0:7])

def get_month(date):
    return int(date[5:7])

def get_color(count, scalar_mappable):
    return scalar_mappable.to_rgba(count)

def get_temp(short_date, temp_data):
    temp_data = temp_data[temp_data['short_date'] == short_date]
    if temp_data.empty:
        return None

    return temp_data['LandAndOceanAverageTemperature'].values[0]

# =============== Disaster decloration ===============

disaster_data = pd.read_csv("../data/disaster_decloration.csv")
disaster_data['year'] = disaster_data['declaration_date'].apply(get_year)
disaster_data['short_date'] = disaster_data['declaration_date'].apply(get_date)
disaster_data = disaster_data[disaster_data['incident_type'] == 'Flood']
disaster_data['fin_date'] = pd.to_datetime(disaster_data['declaration_date'], utc=True)
disaster_data = disaster_data[disaster_data['year'] <= 2015] # When the disaster data starts
# disaster_data = disaster_data[disaster_data['year'] >= 1985] # zoomed in for Figure 3

# =============== Global temps ===============

temp_data = pd.read_csv("../data/global_temp.csv")
temp_data['fin_date'] = pd.to_datetime(temp_data['dt'], utc=True)
temp_data['short_date'] = temp_data['dt'].apply(get_date)
temp_data['year'] = temp_data['dt'].apply(get_year)
temp_data = temp_data[temp_data['year'] >= 1953] # When the disaster data starts

# Attach temp data to each disaster
disaster_data['temperature'] = disaster_data['short_date'].apply(get_temp, args=(temp_data,)).dropna()
grp = disaster_data.groupby('fin_date').count()
disaster_data = disaster_data.join(grp, on='fin_date', rsuffix='_count')

norm = Normalize(vmin=disaster_data['temperature_count'].min(), vmax=disaster_data['temperature_count'].max())
sm = cm.ScalarMappable( norm=norm, cmap='cividis')
disaster_data['color'] = disaster_data['temperature_count'].apply(get_color, args=(sm,))

# Draw dots above grid
plt.rc('axes', axisbelow=True) #https://stackoverflow.com/a/42951885
plt.grid()

# So the largest ones are drawn on top last
disaster_data = disaster_data.sort_values(by='temperature_count')

# https://stackoverflow.com/questions/14827650/pyplot-scatter-plot-marker-size
# https://stackoverflow.com/questions/19064772/visualization-of-scatter-plots-with-overlapping-points-in-matplotlib
plt.scatter(
    disaster_data['fin_date'],
    disaster_data['temperature'],
    s=disaster_data['temperature_count']*4,
    color=disaster_data['color']
)

plt.xlabel('Year')
plt.ylabel('Land/Ocean Avg Temp (c)')
plt.title("Avg Temp vs Year")
cbar = plt.colorbar(sm, ax=plt.gca())

# https://stackoverflow.com/questions/15908371/matplotlib-colorbars-and-its-text-labels
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('# of Floods', rotation=270)

plt.show()
# plt.savefig("heat.png")