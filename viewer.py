import numpy as np
import matplotlib.pyplot as plt

class Plotter():

	def __init__(self, e, g, a):
		self.e = e
		self.g = g
		self.a = a

	def plot_success(self):
	    result = np.load("result.npy").item()
	    stats = np.load("stats.npy").item()
	    colors = list("rgbcmyk")
	    print stats
	    x = result.keys()
	    y = result.values()
	    fig = plt.figure()
	    fig.suptitle('(Epsilon: {} - Gamma: {} - Alpha: {} \n Success Rate: %{})'.format(self.e, self.g, self.a, stats["winning"]), fontsize=14)
	    plt.scatter(x,y,color=colors.pop())
	    plt.xlabel("Trials", fontsize = 14)
	    plt.ylabel("Rewards", fontsize = 14)
	    plt.legend(result.keys())
	    fig.savefig('(Epsilon{}Gamma{}Alpha{}).png'.format(self.e, self.g, self.a), bbox_inches='tight')

	def plot_times(self):
	    times = np.load("times.npy").item()
	    colors = list("rgbcmyk")
	    x = times.keys()
	    y = times.values()
	    fig = plt.figure()
	    fig.suptitle('(Epsilon: {} - Gamma: {} - Alpha: {})'.format(self.e, self.g, self.a), fontsize=14)
	    plt.scatter(x,y,color=colors.pop())
	    plt.xlabel("Time", fontsize = 14)
	    plt.ylabel("Rewards", fontsize = 14)
	    plt.legend(times.keys())
	    fig.savefig('TIMES(Epsilon{}Gamma{}Alpha{}).png'.format(self.e, self.g, self.a), bbox_inches='tight')

	def plot_rewards(self):
	    result = np.load("rewards.npy").item()
	    colors = list("rgbcmyk")
	    x = result.keys()
	    y = result.values()
	    fig = plt.figure()
	    fig.suptitle('(Epsilon: {} - Gamma: {} - Alpha: {})'.format(self.e, self.g, self.a), fontsize=14)
	    plt.scatter(x,y,color=colors.pop())
	    plt.xlabel("Trials", fontsize = 14)
	    plt.ylabel("Positive Rewards %", fontsize = 14)
	    plt.legend(result.keys())
	    fig.savefig('REWARDS(Epsilon{}Gamma{}Alpha{}).png'.format(self.e, self.g, self.a), bbox_inches='tight')