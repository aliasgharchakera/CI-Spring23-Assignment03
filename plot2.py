import pandas as pd
import geopandas as gpd
from SOM import SOM
import numpy as np


# Read in the CSV file using pandas
df = pd.read_csv('countries of the world.csv')
df.dropna(inplace=True)

# Extract the columns you want to use as inputs for the SOM
data = df[["Population", "Area (sq. mi.)", "Pop. Density (per sq. mi.)", "GDP ($ per capita)",	"Literacy (%)"]].to_numpy()
countries = df[["Country"]]["Country"].values.tolist()
countries = [i[:-1] for i in countries]

# Create a SOM object
som = SOM(map_size=(10, 10), input_size=data.shape[1], sigma=2.0, learning_rate=0.1, num_epochs=100, c=True, countries=countries)

# Train the SOM on the data
som.train(data)

# Get the coordinates of the SOM nodes
# coords = np.random.rand(som.map_size[0], som.map_size[1], data.shape[1])
coords = []
for i in range(som.map_size[0]):
    for j in range(som.map_size[1]):
        coords.append((i, j, som.weights[i,j,:]))
        # coords[i][j]

# Create a GeoDataFrame of the countries
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))
# print(coords)
# Add the coordinates of the SOM nodes to the GeoDataFrame
coords_array = np.array(coords)
gdf_nodes = gpd.GeoDataFrame(pd.DataFrame(coords, columns=["x", "y", "weights"]), geometry=gpd.points_from_xy(coords[:,2][:,1], coords[:,2][:,0]))
weights_array = [t[2] for t in coords]
gdf_nodes = gpd.GeoDataFrame(pd.DataFrame(coords, columns=["x", "y", "weights"]), geometry=gpd.points_from_xy(weights_array[:,1], weights_array[:,0]))

# Plot the SOM nodes on the map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
ax = world.plot(color='white', edgecolor='black')
gdf_nodes.plot(ax=ax, column='weights', markersize=20, cmap='viridis')
