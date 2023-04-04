import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SOM import SOM
import geopy.geocoders as geocoders
import geopandas as gpd

# Read in the CSV file using pandas
df = pd.read_csv('countries of the world.csv')
df.dropna(inplace=True)

# Extract the columns you want to use as inputs for the SOM
data = df[["Population", "Area (sq. mi.)", "Pop. Density (per sq. mi.)", "GDP ($ per capita)", "Literacy (%)"]].to_numpy()
countries = df[["Country"]]["Country"].values.tolist()
countries = [i[:-1] for i in countries]

# Geocode the countries to get their coordinates
geolocator = geocoders.Nominatim(user_agent="my_app")
coordinates = [geolocator.geocode(c).point for c in countries]

# Create a SOM object
som = SOM(map_size=(10, 10), input_size=data.shape[1], sigma=2.0, learning_rate=0.1, num_epochs=100, c=True, countries=countries)

# Train the SOM on the data
som.train(data)

# Create a GeoDataFrame with the country names, coordinates, and SOM predictions
gdf = gpd.GeoDataFrame({'Country': countries, 'Coordinates': coordinates, 'SOM_Predictions': som.predict(data)})

# Plot the SOM predictions on a world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(figsize=(10, 6))
gdf.plot(column='SOM_Predictions', cmap='YlOrRd', legend=True, ax=ax)
plt.show()
