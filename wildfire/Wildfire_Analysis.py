# Adam Pywell
# Data Source: https://www.kaggle.com/datasets/joebeachcapital/global-earth-temperatures

from matplotlib import cm
from matplotlib.colors import Normalize
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import linregress
import sys

# Function to get color based on count using a scalar mappable
def get_color(count, scalar_mappable):
    return scalar_mappable.to_rgba(count)

# Read command line arguments for file paths
wildfire_file = sys.argv[1]
temp_file = sys.argv[2]

# Load and filter temperature data
temp_table = pd.read_csv(temp_file)
temp_table = temp_table[temp_table['Year'] > 2011]

# Calculate average monthly temperature anomaly per year
temp_avg = temp_table.groupby('Year').agg({'Monthly Anomaly': 'mean'})

print(temp_avg)

# Load and filter wildfire data
table = pd.read_csv(wildfire_file)
table = table[(table['Year'] > 2011) & (table['Year'] != 2024)]
table = table.rename(columns={'Annual number of fires': 'Wildfire Count'})

# Aggregate wildfire counts by country and sort
country_fires = table.groupby('Entity').agg({'Wildfire Count': 'sum'})
country_fires = country_fires.reset_index()
country_fires = country_fires.sort_values(by='Wildfire Count')
country_fires = country_fires[country_fires['Wildfire Count'] > 0]
country_fires = country_fires.tail(30)
country_fires = country_fires.drop(country_fires.tail(1).index)

# Separate continent and country data
continent_fires = country_fires[country_fires['Entity'].isin(['Europe','Africa','South America', 'Asia','North America', 'Oceania'])]
country_fires = country_fires[~country_fires['Entity'].isin(['Europe','Africa','South America', 'Asia','North America', 'Oceania'])]

# Aggregate annual wildfire counts globally
annual_fires = table.groupby('Year').agg({'Wildfire Count':'sum'}).reset_index()
annual_fires.rename(columns={'Wildfire Count': 'Annual Wildfires Globally'}, inplace=True)

# Join wildfire data with temperature data
fire_join = annual_fires.join(temp_avg['Monthly Anomaly'], on='Year')
fire_join['Relative Count'] = fire_join['Annual Wildfires Globally'] - fire_join['Annual Wildfires Globally'].min()

print(fire_join)
print(country_fires)
print(continent_fires)
print(annual_fires)

# Perform linear regression and correlation analysis
fit = linregress(fire_join['Annual Wildfires Globally'], fire_join['Monthly Anomaly'])
print('Linear Regression Statistics -->', fit)
print('PValue -->', fit.pvalue)
corr = pearsonr(fire_join['Annual Wildfires Globally'], fire_join['Monthly Anomaly'])
print("Correlation Statistic -->", corr.statistic)

# Plot histogram of annual wildfire observations
plt.figure()
plt.title('Frequency of Similar Annual WildFire Observations (2012 to 2023)')
plt.xlabel('Annual Wildfires Observations (Scale = 1,000,000)')
plt.ylabel('Frequency (Count)')
plt.xticks(np.arange(2.4 * 10**6, 3.1 * 10**6, 25000)-0.5, rotation=45)
plt.yticks(np.arange(0,5,1))
plt.hist(annual_fires['Annual Wildfires Globally'], ec='black', alpha=1, align='mid', rwidth=0.3)

# Normalize data for color mapping
norm = Normalize(vmin=fire_join['Relative Count'].min(), vmax=fire_join['Relative Count'].max())
sm = cm.ScalarMappable( norm=norm, cmap='Oranges')
fire_join['color'] = fire_join['Relative Count'].apply(get_color, args=(sm,))

# Plot scatter plot of temperature anomalies vs wildfire observations
plt.figure()
plt.rc('axes', axisbelow=True) # Ensure grid is behind plot elements
plt.grid()
plt.xlabel('Year')
plt.ylabel('Average Temperature Anomaly')
plt.title("Yearly Average Temperature Anomaly vs Yearly Wildfire Observations")
cbar = plt.colorbar(sm, ax=plt.gca())
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('Relative Wildfire Observations', rotation=270)
plt.scatter(fire_join['Year'], fire_join['Monthly Anomaly'], s=fire_join['Relative Count'] / 200, color=fire_join['color'] )

# Plot bar chart of top 23 countries by wildfire observations
plt.figure()
plt.title('Top 23 Total Wildfire Observations (By Country)')
plt.xlabel('Country')
plt.ylabel('Total Wildfire Observations (Last 11 Years)')
plt.tight_layout(pad=0.5)
plt.xticks(rotation=270)
plt.bar(country_fires['Entity'], country_fires['Wildfire Count'], ec='black', color='red')

# Plot bar chart of top continents by wildfire observations
plt.figure()
plt.title('Top 23 Total Wildfire Observations (By Continent)')
plt.xlabel('Continent')
plt.ylabel('Total Wildfire Observations (Last 11 Years)')
plt.tight_layout(pad=0.5)
plt.xticks(rotation=45)
plt.bar(continent_fires['Entity'], continent_fires['Wildfire Count'],ec='black', color='green')
plt.show()