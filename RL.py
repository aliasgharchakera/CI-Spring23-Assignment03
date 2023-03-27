import numpy as np
class GridworldRLAgent:
    def __init__(self, n, T, alpha, gamma):
        self.n = n  # size of gridworld (n x n)
        self.T = T  # initial temperature parameter for Boltzmann distribution
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.Q = np.zeros((n*n, 4))  # state-action value function
        self.V = np.zeros(n*n)  # value function

    def learn(self, num_episodes):
        for episode in range(num_episodes):
            # initialize state
            s = self._reset_state()

            # initialize episode variables
            G = 0  # return
            t = 0  # time step
            done = False

            while not done:
                # select action using Boltzmann distribution
                a = self._select_action(s, self.T)

                # take action and observe reward and next state
                s_prime, r, done = self._step(s, a)

                # calculate TD error and update Q value
                td_error = r + self.gamma * np.max(self.Q[s_prime]) - self.Q[s, a]
                self.Q[s, a] += self.alpha * td_error

                # update return
                G += r * self.gamma**t

                # update state
                s = s_prime

                # increment time step
                t += 1

            # update temperature parameter
            self.T *= 0.99

            # update value function
            self.V = np.max(self.Q, axis=1)

    def evaluate(self, num_episodes):
        rewards = []
        for episode in range(num_episodes):
            # initialize state
            s = self._reset_state()

            # initialize episode variables
            G = 0  # return
            t = 0  # time step
            done = False

            while not done:
                # select action using greedy policy
                a = self._select_greedy_action(s)

                # take action and observe reward and next state
                s_prime, r, done = self._step(s, a)

                # update return
                G += r * self.gamma**t

                # update state
                s = s_prime

                # increment time step
                t += 1

            # append episode reward
            rewards.append(G)

        # print average reward
        print("Average reward:", np.mean(rewards))

    def print_value_function(self):
        print("Value function:")
        for i in range(self.n):
            for j in range(self.n):
                s = i*self.n + j
                print("{:.2f}".format(self.V[s]), end=" ")
            print()


# create agent
agent = GridworldRLAgent(n=5, T=1.0, alpha=0.1, gamma=0.9)

# learn for 1000 episodes
agent.learn(num_episodes=1000)

# evaluate for 100 episodes
agent.evaluate(num_episodes=100)

# print value function
agent.print_value_function()
