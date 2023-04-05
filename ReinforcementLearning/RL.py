from state import State
import random
from vis import DisplayGrid
import math

class RL:

    def __init__(self, size, redTerminals, greenTerminals, episodes, gamma = 0.2, alpha = 0.6, k = 0.6):
        '''
        Constructor for the RL. 
        size: size of the grid
        redTerminals: a list containing coordinates for the red terminals
        greenTerminals: a list containing coordinates for the green terminals
        alpha = the learning rate for agent
        gamma = the discount factor for the agent 
        '''
        self.gamma = gamma
        self.alpha = alpha
        self.k = k
        self.size = size
        self.grid = [[State() for j in range(self.size)] for i in range(self.size)]
        self.currentState = self.getStartPos()
        self.redTerminals = redTerminals
        self.greenTerminals = greenTerminals
        self.episodes = episodes
        self.setupTerminal()

    def accessGridState(self,i,j):
        '''
        checks if any cell in the grid is a terminal or not.
        '''
        return self.grid[i][j].isTerminal

    def setupTerminal(self):
        '''
        Given the coordinates, it places red and green terminals on the grid.
        '''
        for i in self.redTerminals:
            self.grid[i[0]][i[1]].isTerminal = True
            self.grid[i[0]][i[1]].value = -100
            self.grid[i[0]][i[1]].reward = -100

        for x in self.greenTerminals:
            self.grid[x[0]][x[1]].isTerminal = True
            self.grid[x[0]][x[1]].reward = 100
            self.grid[x[0]][x[1]].value = 100
        
    def drawGrid(self):
        '''
        draws the grid on the terminal  
        '''
        for i in self.grid:
            for j in i:
                print(round(j.value,2), end = "  ")
            print()
        
    def getStartPos(self):
        '''
        Gives a random start position for the agent.
        '''
        i = random.randint(0,self.size - 1)
        j = random.randint(0,self.size - 1)
        while self.accessGridState(i, j) == True:
            i = random.randint(0,self.size - 1)
            j = random.randint(0,self.size - 1)
        return (i,j)
    
    def decreaseTemperature(self,episode:int):
        '''
        Decreases temperature after each episode. 
        '''
        if (1 - (self.k/episode)) < 0.4:
            self.k = 0.4
        else:
            self.k = 1 - round(self.k/episode,1)


    def getAction(self) -> str:
        '''
        Making use of Boltzmann Distribution Function, it assigns a probability to each up, down, left and right moves. 
        '''
        i, j = self.currentState
        actions = {}
        if i == 0:
            if j == self.size - 1:
                actions['up'] = 0
                actions['down'] = math.exp(self.grid[i+1][j].value/self.k) / (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i + 1][j].value/self.k))
                actions['left'] = math.exp(self.grid[i][j-1].value/self.k) / (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i + 1][j].value/self.k))
                actions['right'] = 0
            elif j == 0:
                actions['up'] = 0
                actions['down'] = math.exp(self.grid[i+1][j].value/self.k) /  (math.exp(self.grid[i][j+1].value/self.k) + math.exp(self.grid[i + 1][j].value/self.k))
                actions['left'] = 0
                actions['right'] = math.exp(self.grid[i][j+1].value/self.k) / (math.exp(self.grid[i][j+1].value/self.k) + math.exp(self.grid[i + 1][j].value/self.k))
            else:
                actions['up'] = 0
                actions['down'] = math.exp(self.grid[i+1][j].value/self.k) / (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i + 1][j].value/self.k)  + math.exp(self.grid[i][j+1].value/self.k))
                actions['left'] = math.exp(self.grid[i][j-1].value/self.k) / (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i + 1][j].value/self.k)  + math.exp(self.grid[i][j+1].value/self.k))
                actions['right']= math.exp(self.grid[i][j+1].value/self.k) / (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i + 1][j].value/self.k)  + math.exp(self.grid[i][j+1].value/self.k))

        elif i == self.size - 1:
            if j == self.size - 1:
                actions['up'] =   math.exp(self.grid[i-1][j].value/self.k) / (math.exp(self.grid[i][j-1].value/self.k)  + math.exp(self.grid[i-1][j].value/self.k))
                actions['down'] =  0
                actions['left'] = math.exp(self.grid[i][j-1].value/self.k) / (math.exp(self.grid[i][j-1].value/self.k)  + math.exp(self.grid[i-1][j].value/self.k))
                actions['right'] = 0

            elif j == 0:
                actions['up'] = math.exp(self.grid[i-1][j].value/self.k)    / (math.exp(self.grid[i][j+1].value/self.k)  + math.exp(self.grid[i-1][j].value/self.k))
                actions['down'] = 0
                actions['left'] = 0
                actions['right'] = math.exp(self.grid[i][j+1].value/self.k) / (math.exp(self.grid[i][j+1].value/self.k)  + math.exp(self.grid[i-1][j].value/self.k))
            else:
                actions['up'] = math.exp(self.grid[i-1][j].value/self.k) /   (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i-1][j].value/self.k)  + math.exp(self.grid[i][j+1].value/self.k))
                actions['down'] = 0
                actions['left'] = math.exp(self.grid[i][j-1].value/self.k) / (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i-1][j].value/self.k)  + math.exp(self.grid[i][j+1].value/self.k))
                actions['right'] = math.exp(self.grid[i][j+1].value/self.k) /(math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i-1][j].value/self.k)  + math.exp(self.grid[i][j+1].value/self.k))
        
        else:
            if j == self.size - 1:
                actions['up'] = math.exp(self.grid[i-1][j].value/self.k) /   (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i-1][j].value/self.k)  + math.exp(self.grid[i+1][j].value/self.k))
                actions['down'] = math.exp(self.grid[i+1][j].value/self.k) / (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i-1][j].value/self.k)  + math.exp(self.grid[i+1][j].value/self.k))
                actions['left'] = math.exp(self.grid[i][j-1].value/self.k) / (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i-1][j].value/self.k)  + math.exp(self.grid[i+1][j].value/self.k))
                actions['right'] = 0
            elif j == 0:
                actions['up'] = math.exp(self.grid[i-1][j].value/self.k) /    (math.exp(self.grid[i-1][j].value/self.k) + math.exp(self.grid[i][j+1].value/self.k)  + math.exp(self.grid[i+1][j].value/self.k))
                actions['down'] = math.exp(self.grid[i+1][j].value/self.k) /  (math.exp(self.grid[i-1][j].value/self.k) + math.exp(self.grid[i][j+1].value/self.k)  + math.exp(self.grid[i+1][j].value/self.k))
                actions['left'] = 0
                actions['right'] = math.exp(self.grid[i][j+1].value/self.k) / (math.exp(self.grid[i-1][j].value/self.k) + math.exp(self.grid[i][j+1].value/self.k)  + math.exp(self.grid[i+1][j].value/self.k))
            else:
                actions['up'] = math.exp(self.grid[i-1][j].value/self.k) /    (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i-1][j].value/self.k) + math.exp(self.grid[i][j+1].value/self.k)  + math.exp(self.grid[i+1][j].value/self.k))
                actions['down'] = math.exp(self.grid[i+1][j].value/self.k) /  (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i-1][j].value/self.k) + math.exp(self.grid[i][j+1].value/self.k)  + math.exp(self.grid[i+1][j].value/self.k))
                actions['left'] = math.exp(self.grid[i][j-1].value/self.k) /  (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i-1][j].value/self.k) + math.exp(self.grid[i][j+1].value/self.k)  + math.exp(self.grid[i+1][j].value/self.k))
                actions['right'] = math.exp(self.grid[i][j+1].value/self.k) / (math.exp(self.grid[i][j-1].value/self.k) + math.exp(self.grid[i-1][j].value/self.k) + math.exp(self.grid[i][j+1].value/self.k)  + math.exp(self.grid[i+1][j].value/self.k))

        #Making use of fitness proportional share, it selects the next state based on the given probabilities.
        ranges = []
        start = 1
        for i in actions:
            end = start + actions[i]
            ranges.append((start, end))
            start = end

        low = ranges[0][0]
        high = ranges[len(ranges)-1][1]
        randomNum = random.uniform(low,high)
        for i in range(len(ranges)):
            if randomNum > ranges[i][0] and randomNum <= ranges[i][1]:
                if i == 0:
                    return 'up'
                elif i == 1:
                    return 'down'
                elif i == 2:
                    return 'left'
                else:
                    return 'right'
        

    def runEpisode(self):
        '''
        Starts the episodic learning. 
        '''
        for i in range(self.episodes):    
            print("Episode: ", i)
            self.currentState = self.getStartPos()
            x,y = self.currentState

            while not self.grid[self.currentState[0]][self.currentState[1]].isTerminal:
                action = self.getAction()
                if action == 'left':
                    i = self.currentState[0]
                    j = self.currentState[1] - 1

                elif action == 'right':
                    i = self.currentState[0]
                    j = self.currentState[1] + 1

                elif action == 'up':
                    i = self.currentState[0] - 1
                    j = self.currentState[1]

                elif action == 'down':
                    i = self.currentState[0] + 1
                    j = self.currentState[1] 

                #updates the current state.                
                self.grid[self.currentState[0]][self.currentState[1]].value = self.grid[self.currentState[0]][self.currentState[1]].value + (self.alpha)*(self.grid[i][j].reward + (self.gamma)*(self.grid[i][j].value) - self.grid[self.currentState[0]][self.currentState[1]].value)
                self.currentState = (i,j)
            if i != 0:
                self.decreaseTemperature(i)
                # print('Action Selected: ', action)
                # print(self.currentState)
            
    def visualizeGrid(self):
        '''
        Places arrows on the grid. Visualizes Grid on the tkinter screen.
        '''
        arrowDic = {}
        for i in range(self.size):
            for j in range(self.size):
                if self.grid[i][j].isTerminal:
                    pass
                possValues = {}
                if i == 0:
                    if j == 0:
                        possValues['right'] = self.grid[i][j+1].value
                        possValues['down'] = self.grid[i+1][j].value
                    elif j == self.size - 1:
                        possValues['left'] = self.grid[i][j-1].value
                        possValues['down'] = self.grid[i+1][j].value
                    else:
                        possValues['left'] =  self.grid[i][j-1].value
                        possValues['down'] =  self.grid[i+1][j].value
                        possValues['right'] = self.grid[i][j+1].value
                elif i == self.size - 1:
                    if j == self.size - 1:
                        possValues['left'] = self.grid[i][j-1].value
                        possValues['up'] =   self.grid[i-1][j].value
                    elif j == 0:
                        possValues['right'] = self.grid[i][j+1].value
                        possValues['up'] =    self.grid[i-1][j].value
                    else:
                        possValues['left'] =  self.grid[i][j-1].value
                        possValues['up'] =    self.grid[i-1][j].value
                        possValues['right'] = self.grid[i][j+1].value
                else:
                    if j == self.size - 1:
                        possValues['left'] = self.grid[i][j-1].value
                        possValues['up'] =   self.grid[i-1][j].value
                        possValues['down'] = self.grid[i+1][j].value
                    elif j == 0:
                        possValues['down'] =  self.grid[i+1][j].value
                        possValues['right'] = self.grid[i][j+1].value
                        possValues['up'] =    self.grid[i-1][j].value
                    else:
                        possValues['left'] =  self.grid[i][j-1].value
                        possValues['up'] =    self.grid[i-1][j].value
                        possValues['right'] = self.grid[i][j+1].value
                        possValues['down'] =  self.grid[i+1][j].value
                possValues = dict(sorted(possValues.items(), key=lambda x: x[1], reverse=True))
                first = next(iter(possValues))
                arrowDic[(i,j)] = first

        grid = DisplayGrid(self.size, self.redTerminals, self.greenTerminals, arrowDic)
        grid.createWindow()


