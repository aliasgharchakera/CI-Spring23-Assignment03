import numpy as np
import matplotlib.pyplot as plt
import geopandas

class SOM:
    def __init__(self, mapSize, inputSize, sigma, learningRate, numEpochs, c=False, countries=None):
        self.mapSize = mapSize
        self.inputSize = inputSize
        self.sigma = sigma
        self.learningRate = learningRate
        self.numEpochs = numEpochs
        self.c = c
        self.weights = np.random.rand(mapSize[0], mapSize[1], inputSize)
        if self.c:
            self.countries = countries
            self.countryMap = {}
            for country in countries:
                self.countryMap[country] = (0, 0)
        self.rgb = dict()
        # print(self.weights)

    def getDistance(self, x, y):
        # Calculate the Euclidean distance between two vectors
        return np.linalg.norm(x - y)

    def getBmu(self, x):
        # Find the best matching unit for a given vector
        distances = np.zeros((self.mapSize[0], self.mapSize[1]))
        for i in range(self.mapSize[0]):
            for j in range(self.mapSize[1]):
                distances[i,j] = self.getDistance(x, self.weights[i,j])
        bmu = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu

    def getNeighborhood(self, bmu, sigma):
        # Calculate the neighborhood function for a given BMU
        neighborhood = np.zeros((self.mapSize[0], self.mapSize[1]))
        for i in range(self.mapSize[0]):
            for j in range(self.mapSize[1]):
                neighborhood[i,j] = np.exp(-((i-bmu[0])**2 + (j-bmu[1])**2)/(2*sigma**2))
        return neighborhood

    def updateWeights(self, x, bmu, neighborhood, learningRate):
        # Update the weights of the SOM
        for i in range(self.mapSize[0]):
            for j in range(self.mapSize[1]):
                self.weights[i,j,:] += learningRate * neighborhood[i,j] * (x - self.weights[i,j,:])
        # self.weights /= self.weights.max(axis=0)

    def train(self, data):
        # Train the SOM
        for epoch in range(self.numEpochs):
            self.bmu = list()
            for index, point in enumerate(data):
                bmu = self.getBmu(point)
                self.bmu.append(bmu)
                if self.c:
                    self.countryMap[self.countries[index]] = bmu
                neighborhood = self.getNeighborhood(bmu, self.sigma)
                self.updateWeights(point, bmu, neighborhood, self.learningRate)
            self.sigma *= 0.9
            self.learningRate *= 0.9
    
    def color(self, var1, var2):
        figure = plt.figure(figsize=(self.mapSize[0], self.mapSize[1]))
        ax = figure.add_subplot(111, aspect='equal')
        ax.set_xlim((0, self.mapSize[0]))
        ax.set_ylim((0, self.mapSize[1]))
        plt.rcParams.update({'font.size': 5})
        for row in range(self.mapSize[0]):
            for column in range(self.mapSize[1]):
                weight = self.weights[row][column]
                rgb = np.zeros(3)
                for i in range(len(weight)):
                    if i % 3 == 0:
                        rgb[0] += weight[i] * var1
                    elif i % 3 == 1:
                        rgb[1] += weight[i] * var1
                    else:
                        rgb[2] += weight[i] * var1

                # Normalizing the rgb values
                rgb = rgb / rgb.sum()
                self.rgb[(row, column)] = rgb
                ax.add_patch(plt.Rectangle((row, column), 1, 1, facecolor=(
                rgb[0], rgb[1], rgb[2], 1), edgecolor='black'))
                # plt.show()
        grid = []
        self.countryColor = {}
        for i in range(len(self.bmu)):
            country = self.countries[i]
            bmu = self.bmu[i]
            self.countryColor[country] = self.rgb[bmu[0], bmu[1]]
            x = bmu[0] + var2
            y = bmu[1] + var2
            counter = 0
            while (x, y) in grid and counter < 4:
                y = y + var2
                counter += 1
            grid.append((x, y))
            plt.text(x, y, country)
            
        world = geopandas.read_file(geopandas.datasets.get_path('naturalearth_lowres'))
        fig, ax = plt.subplots(figsize=(10, 10))
        world.plot(ax=ax, facecolor='lightgray', edgecolor='black')
        for i in self.countryColor:
            color = self.countryColor[i]
            # print(color, i)
            if i in world["name"].tolist():
                world[world.name == i].plot(color=color, ax=ax)
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title('Self Organizing Map Visualization with Countries')
        plt.show()
