import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import numpy as np


class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        # TODO: Initialize any additional variables here
        self.actions = [None, 'forward', 'left', 'right']
        self.oldState = ""
        self.oldReward = 0
        self.oldAction = None
        self.epsilon = 0.05
        self.gamma = 0.7
        self.alpha = 0.3
        self.qtable = dict()
        #self.qtable = np.load('my_file.npy').item()



    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)

        # TODO: Update stat
        ocg = inputs["oncoming"]
        if ocg == None:
            ocg="None"

        left = inputs["left"]
        if left == None:
            left = "None"

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        keys = self.qtable.keys()
        self.future_rewards = -1000
        for x in keys:
            #print x, self.qtable[x]
            if x[0] == self.oldState and (self.qtable[x] > self.future_rewards):
                self.future_rewards = self.qtable[x]
        #self.qtable[(self.oldState, self.oldAction)]= (1-self.alpha)*self.oldReward + self.alpha * ( self.gamma * self.future_rewards )
        #self.qtable[(self.oldState, self.oldAction)] = self.oldReward + self.future_rewards
        self.qtable[(self.oldState, self.oldAction)]= (1-self.alpha)*
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

        state = inputs["light"]+","+ocg+","+left+","+self.next_waypoint
        self.oldState = state
        self.state = state
        # TODO: Select action according to your policy+

        action = self.optimal_action(state)
        self.oldAction = action
        # Execute action and get reward
        reward = self.env.act(self, action)
        self.oldReward = reward

        print "\n"
        # TODO: Learn policy based on state, action, reward

        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]
        np.save('my_file.npy', self.qtable)

    def optimal_action(self, state):

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


def run():
    """Run the agent for a finite number of trials."""

    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0.0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False

    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()
