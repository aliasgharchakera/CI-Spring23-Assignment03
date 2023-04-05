class State:
    
    def __init__(self, value = 0, reward = -1, isTerminal = False):
        self.value = value
        self.reward = reward
        self.isTerminal = isTerminal
