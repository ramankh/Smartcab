import numpy as np
import matplotlib.pyplot as plt


stat = np.load("result.npy").item()


colors = list("rgbcmyk")

x = stat.keys()
y = stat.values()
plt.scatter(x,y,color=colors.pop())
plt.xlabel("Trials", fontsize = 14)
plt.ylabel("Rewards", fontsize = 14)
plt.legend(stat.keys())
plt.show()