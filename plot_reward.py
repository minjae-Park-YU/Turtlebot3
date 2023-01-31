import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

file_names = ['reward_ep100.csv']
all_data = []
for name in file_names:
    dataset = pd.read_csv(name)
    for data in dataset['reward']:
        all_data.append(data)

x = np.arange(1, 102)

# rolling_means = []
# N = 150
# for i in range(len(all_data)):
#     list1 = all_data[i]
#     rolling_mean = [np.mean(list1[k:k+N]) for k in range(len(list1)-N+1)]
#     rolling_means.append(rolling_mean)

plt.plot(x, all_data, '-b', linewidth='0.5', label='base_code')
# plt.plot(x, rolling_means[1], '-b', linewidth='0.5', label='experiment 4')
# plt.plot(x, rolling_means[2], '-g', linewidth='0.5', label='100 epi, vir 9')
# plt.plot(x, rolling_means[3], '-b', linewidth='0.5', label='500 epi, vir 9')
# plt.plot(x, rolling_means[4], 'pink', linewidth='0.5', label='500 epi, vir 20')
# plt.plot(x, rolling_means[5], '-y', linewidth='0.5', label='virtual 9')
plt.legend()
plt.show()
