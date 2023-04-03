import pandas as pd
import numpy as np
# from SOM import SOM

# Read in the CSV file using pandas
df = pd.read_csv('API_11_DS2_en_csv_v2_5351544/API_11_DS2_en_csv_v2_5351544.csv')

# Extract the columns you want to use as inputs for the SOM
data = df[['Country Name', '2018', '2019', '2020']].to_numpy()

print(data)
# # Create a SOM object
# som = SOM(map_size=(10, 10), input_size=data.shape[1]-1, sigma=5.0, learning_rate=0.5, num_epochs=100)

# # Train the SOM on the data
# som.train(data[:,1:])

# # Get the cluster labels for the data
# cluster_labels = som.predict(data[:,1:])

# # Add the cluster labels as a new column in the data
# df['Cluster'] = cluster_labels

# # Do something with the data
# print(df.head())
