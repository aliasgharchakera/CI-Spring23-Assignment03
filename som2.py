import matplotlib.pyplot as plt
import geopands as gpd

def somGrid(self, activated):
        '''
        This function is used to plot the grid of the self organizing map

        There are few of the major changes taken from ChatGPT's code. The link to his code is given below in the References Section: 


        parameters:
        - self : mandatory parameter for all the functions in python
        - activated : the activation function that is being used

        returns:
        - None
        '''

        self.colorGrid = {}
        figure = plt.figure(figsize=(self.height, self.width))
        ax = figure.add_subplot(111, aspect='equal')
        plt.rcParams.update({'font.size': 6})

        for row in range(self.height):
            for column in range(self.width):
                temp = row*self.width + column
                weights = self.neurons[temp].weights
                rgb = [0, 0, 0]
                ax.set_xlim((0, self.width))
                ax.set_ylim((0, self.height))
                for i in range(len(weights)):
                    if i % 3 == 0:
                        rgb[0] = rgb[0] + weights[i] * activated
                    elif i % 3 == 1:
                        rgb[1] = rgb[1] + weights[i] * activated
                    else:
                        rgb[2] = rgb[2] + weights[i] * activated

                # Normalizing the rgb values
                rgbSum = sum(rgb)
                for i in range(len(rgb)):
                    rgb[i] = rgb[i]/rgbSum
                self.rgb = rgb
                # Plotting the grid
                ax.add_patch(plt.Rectangle((row, column), 1, 1, facecolor=(
                    rgb[0], rgb[1], rgb[2], 1), edgecolor='black'))
                self.colorGrid[(row, column)] = rgb

def colorMap(self, activated):
    '''
    This function is used to plot the color map of the self organizing map with the countries

    There are few of the major changes taken from ChatGPT's code. The link to his code is given below in the References Section:

    parameters:
    - self : mandatory parameter for all the functions in python

    returns:
    - None
    '''

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

def mapVisualization(self):
    '''
    This function is used to plot the worldMap map with the countries

    There are few of the major changes taken from ChatGPT's code. The link to his code is given below in the References Section:

    parameters:
    - self : mandatory parameter for all the functions in python

    returns:
    - None
    '''

    worldMap = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    fig, ax = plt.subplots(figsize=(10, 10))
    worldMap.plot(ax=ax, facecolor='lightgray', edgecolor='black')
    for i in self.colourMatch:
        color = self.colourMatch[i]
        if i in worldMap["iso_a3"].tolist():
            worldMap[worldMap.iso_a3 == i].plot(color=color, ax=ax)
    plt.show()