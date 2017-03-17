import matplotlib.pyplot as plt
import numpy as np

train = np.array([0.2, 0.4, 0.6, 0.8, 1])
# train *= 5652

plt.plot(train, [6.04871, 5.90008, 5.86151, 5.81400, 5.82368])
plt.ylabel('rmse')
plt.xlabel('ratio of training data')
plt.savefig('plot.png')
