import pandas as pd
from SOM.SOM import SOM

def readData(path):
    # Read in the CSV file using pandas
    df = pd.read_csv(path)
    # Drop any rows that have missing data
    df.dropna(inplace=True)
    # Extract the columns you want to use as inputs for the SOM
    data = df[["Population", "Area (sq. mi.)", "GDP ($ per capita)", "Literacy (%)", "Phones (per 1000)", "Pop. Density (per sq. mi.)", "Agriculture", "Industry", "Service"]].to_numpy()
    # Convert the countries to a list
    countries = df[["Country"]]["Country"].values.tolist()
    return data, countries
data, countries = readData('countries of the world.csv')
# Create a SOM object
som = SOM(mapSize=(10, 10), inputSize=data.shape[1], sigma=2.0, learningRate=0.1, numEpochs=100, c=True, countries=countries)

# Train the SOM on the data
som.train(data)
# Plot the results
som.color(0.7, 0.2)