import pandas as pd
import geopandas as gpd
from SOM import SOM
import numpy as np

# Read in the CSV file using pandas
df = pd.read_csv('countries of the world.csv')
df.dropna(inplace=True)

# Extract the columns you want to use as inputs for the SOM
data = df[["Population", "Area (sq. mi.)", "Pop. Density (per sq. mi.)", "GDP ($ per capita)", "Literacy (%)"]].to_numpy()
countries = df[["Country"]]["Country"].values.tolist()
countries = [i[:-1] for i in countries]

# Create a SOM object
som = SOM(map_size=(10, 10), input_size=data.shape[1], sigma=2.0, learning_rate=0.1, num_epochs=100, c=True, countries=countries)

# Train the SOM on the data
som.train(data)

# Get the coordinates of the SOM nodes
# coords = np.zeros((som.map_size[0] * som.map_size[1], 3))
# count = 0
# for i in range(som.map_size[0]):
#     for j in range(som.map_size[1]):
#         coords[count] = [i, j, som.weights[i,j,:][::-1]]
#         count += 1

# Create a GeoDataFrame of the countries
# gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.Longitude, df.Latitude))

# # Add the coordinates of the SOM nodes to the GeoDataFrame
# gdf_nodes = gpd.GeoDataFrame(pd.DataFrame(coords, columns=["x", "y", "weights"]), geometry=gpd.points_from_xy(coords[:,1], coords[:,0]))

# Plot the SOM nodes on the map

# create an empty geopandas dataframe
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
world['count'] = 0

# create the SOM object and train it
# ...

# create a numpy array to hold the SOM weights and their locations
coords = np.zeros((som.map_size[0] * som.map_size[1], 3))

# fill in the numpy array with the SOM weights and their locations
count = 0
for i in range(som.map_size[0]):
    for j in range(som.map_size[1]):
        coords[count, 0] = i
        coords[count, 1] = j
        coords[count, 2] = som.weights[i, j, :][::-1]
        count += 1

# create a geopandas dataframe from the numpy array and plot it on a map
sommapped = gpd.GeoDataFrame(world.join(pd.DataFrame(coords, columns=["X", "Y", "SOM"])))
sommapped.plot(column='SOM')
