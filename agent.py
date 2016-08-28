import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np
import matplotlib.pyplot as plt
from viewer import Plotter


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""
    posNegRatio = dict()
    def __init__(self, env, epsilon, gamma, alpha):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.actions = [None, 'forward', 'left', 'right']
        self.oldState = ""
        self.oldReward = 0
        self.oldAction = None
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha
        self.qtable = dict()

    def reset(self, destination=None):
        self.planner.route_to(destination)

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # Create state with selected features: oncoming
        # left, light and next_waypoint

        state = (inputs["light"], inputs["oncoming"],  inputs["left"], self.next_waypoint)
        self.state = state

        # Select action according to policy using choose_action function
        action = self.choose_action(state)

        # Execute action and get reward
        reward = self.env.act(self, action)

        self.update_qtable(reward)

        self.oldReward = reward
        self.oldState = state
        self.oldAction = action
        print "\n LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        np.save('my_file.npy', self.qtable)

    def choose_action(self, state):

        print self.epsilon
        reward = -1000
        action = "Random"
        if random.random() > self.epsilon:
            for x in self.qtable.keys():
                if x[0] == state and self.qtable[x] > reward:
                    reward = self.qtable[x]
                    action = x[1]
            if action == "Random":
                action = random.choice(self.actions)
        else:
             action = random.choice(self.actions)
        return action

    def update_qtable(self, reward):
        keys = self.qtable.keys()
        self.future_rewards = -1000
        for x in keys:
            if x[0] == self.oldState and (self.qtable[x] > self.future_rewards):
                self.future_rewards = self.qtable[x]
        if (self.oldState, self.oldAction) not in self.qtable:
            self.qtable[(self.oldState, self.oldAction)] = 0
        self.qtable[(self.oldState, self.oldAction)]= (1-self.alpha)*self.qtable[(self.oldState, self.oldAction)] + self.alpha*(self.oldReward+self.gamma*self.future_rewards)


def run(eps, gamma, alpha):
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent, epsilon = eps, gamma = gamma, alpha = alpha)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line
    pltr = Plotter(eps, gamma, alpha)
    pltr.plot_success()
    pltr.plot_rewards()


def run_multiple():
    for e in [float(j)/100 for j in range(5,7)]:
        for g in [float(j)/10 for j in range(7,9)]:
            for a in [float(j)/10 for j in range(2,4)]:
                run(e, g, a)

if __name__ == '__main__':
    run_multiple()