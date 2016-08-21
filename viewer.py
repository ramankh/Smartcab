import numpy as np
import matplotlib.pyplot as plt


result = np.load("result.npy").item()
stats = np.load("stats.npy").item()

colors = list("rgbcmyk")
print stats
x = result.keys()
y = result.values()
plt.scatter(x,y,color=colors.pop())
plt.xlabel("Trials", fontsize = 14)
plt.ylabel("Rewards", fontsize = 14)
plt.legend(result.keys())
plt.show()