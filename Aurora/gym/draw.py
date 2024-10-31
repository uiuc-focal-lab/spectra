import matplotlib.pyplot as plt
import numpy as np

#rec_k5 = np.load("records_k5.npy")
#rec_k3 = np.load("records_k3.npy")
rec_k10 = np.load("records_k10.npy")
#rec_k1 = np.load("records_k1.npy")
#rec_tensor = np.load("records.npy")
number = rec_k10.shape[0]
print(number)
x_ax = np.arange(0, number, 1)
#plt.plot(x_ax, rec_k1[:number], label='k=1')

#plt.plot(x_ax, rec_k5[:number], label='k=5')
plt.plot(x_ax, rec_k10[:number], label='k=10, pytorch')
#plt.plot(x_ax, rec_tensor[:number], label='k=10, tensorflow')
plt.xlabel("Training Episode")
plt.ylabel("Training Reward")
plt.legend()
plt.ylim(ymin=0)
plt.show()
