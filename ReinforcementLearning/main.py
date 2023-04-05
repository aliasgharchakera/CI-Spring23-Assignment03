from RL import RL
import random

def createRandomTerminal(size:int, redCount:int, greenCount:int):
    '''
    Creates random red and green terminals based on the grid size.
    size: size of the grid
    redCount: number of red terminals
    greenCount: number of green terminals.
    '''
    redTerminals = []
    greenTerminals = []
    for k in range(redCount):
        i = random.randint(0, size-1)
        j = random.randint(0, size-1)
        while (i,j) in redTerminals:
            i = random.randint(0, size-1)
            j = random.randint(0, size-1)
        redTerminals.append((i,j))
    
    for k in range(greenCount):
        i = random.randint(0, size-1)
        j = random.randint(0, size-1)
        while (i,j) in redTerminals:
            i = random.randint(0, size-1)
            j = random.randint(0, size-1)
        greenTerminals.append((i,j))

    return redTerminals, greenTerminals

def main():
    '''
    Parameters of the GridWorld are defined here. 
    size: size of the grid. 
    redTerminals = the obstacles in the grid 
    greenTerminals = the point where the agents wants to go. 
    episodes = how many episodes to run, as we are doing episodic learning
    agent = the agent in the RL
    '''
    size = 10                                       #change the size of grid here
    redTerminals = [(2, 4), (2, 5), (2, 6), (2, 7), (2, 9),(6, 2), (7, 2), (8, 2), (9, 2), (7, 3),(7, 6), (7, 6), (7, 8),(7,7), (9, 3), (9, 4), (9, 5), (9, 6), (9, 7), (9, 8)] 
    greenTerminals = [(9,9)]
    episodes = 100                                   #change number of episodes here
    # terminals = createRandomTerminal(size,80,5)    #uncomment this line for random red terminals and green terminals 
    # redTerminals = terminals[0]                    #uncomment this line for random red terminals
    # greenTerminals = terminals[1]                  #uncomment this line for random green terminals
    agent = RL(size, redTerminals, greenTerminals,episodes)
    # agent.drawGrid()                               #uncomment this line to see grid on terminal
    agent.runEpisode()
    # agent.drawGrid()                               #uncomment this line to see the final grid with values
    agent.visualizeGrid()

main()