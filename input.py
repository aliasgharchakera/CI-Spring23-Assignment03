import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from SOM import SOM

# Read in the CSV file using pandas
df = pd.read_csv('countries of the world.csv')
df.dropna(inplace=True)

# Extract the columns you want to use as inputs for the SOM
data = df[["Population", "Area (sq. mi.)", "Pop. Density (per sq. mi.)", "GDP ($ per capita)",	"Literacy (%)"]].to_numpy()
countries = df[["Country"]]["Country"].values.tolist()
countries = [i[:-1] for i in countries]
# data = df[["Population", "Area (sq. mi.)"]].to_numpy()
# print(data.shape)
# print(data)
# Create a SOM object
som = SOM(map_size=(10, 10), input_size=data.shape[1], sigma=2.0, learning_rate=0.1, num_epochs=100, c=True, countries=countries)

# Train the SOM on the data
# som.train(data[:,1:])
som.train(data)
# print(som.countryMap)
print(som.weights)
# Get the cluster labels for the data
# cluster_labels = som.predict(data[:,1:])
# cluster_labels = som.predict(data)
# print(cluster_labels)
# print(cluster_labels.shape, countries.shape)
# print(countries)
# Add the cluster labels as a new column in the data
# df['Cluster'] = cluster_labels

# Do something with the data
# print(df)

gridData = []
self.colourMatch = {}
for i in range(len(self.inputData)):
    winNeuron = self.winningNeuron(self.inputData[i])
    self.winNeuronList.append(winNeuronData(
        winNeuron, self.df.loc[i, "Country"]))

for i in range(len(self.winNeuronList)):
    countryName = self.winNeuronList[i].countryName
    winNeuron = self.winNeuronList[i].winNeuron
    self.colourMatch[countryName] = self.colorGrid[winNeuron.xloc,
                                                    winNeuron.yloc]
    centerx = winNeuron.xloc + activated
    centery = winNeuron.yloc + activated
    counter = 0
    while (centerx, centery) in gridData and counter < 4:
        centery = centery + activated
        counter += 1
    gridData.append((centerx, centery))
    plt.text(centerx, centery, countryName)
plt.xlabel('Width')
plt.ylabel('Height')
plt.title('Self Organizing Map Grid View Visualization with Countries')
plt.show()
