import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import tkinter
from tkinter import filedialog
os.environ['KMP_DUPLICATE_LIB_OK']='True'

root = tkinter.Tk()
root.withdraw()
dir_path = filedialog.askdirectory(parent=root,initialdir="/home/park/anaconda3/envs/2048RL/Codes/",title='Please select a directory')
dir_path = dir_path + "/*"
file_list = glob.glob(dir_path)
file_names_csv = [file for file in file_list if file.endswith(".csv")]

N = 10  # Rolling mean 간격 설정
datasets = []
data_min = []
data_max = []
data_mean = []
for i in range(len(file_names_csv)):
    datasets.append(pd.read_csv(file_names_csv[i]))

for j in range(len(datasets[0]['reward'])):
    temp_list = []
    for k in range(len(file_names_csv)):
        temp_list.append(datasets[k]['reward'][j])
    data_min.append(np.min(temp_list))
    data_max.append(np.max(temp_list))
    data_mean.append(np.mean(temp_list))


list1 = data_min
rolling_means_min = [np.mean(list1[p:p+N]) for p in range(len(list1)-N+1)]
list2 = data_max
rolling_means_max = [np.mean(list2[p:p+N]) for p in range(len(list2)-N+1)]
list3 = data_mean
rolling_means_mean = [np.mean(list3[p:p+N]) for p in range(len(list3)-N+1)]

x = np.arange(1, len(datasets[0]['reward'])-N+2)

plt.plot(x, rolling_means_min, '-k', linewidth='0.5', alpha=0.3)
plt.plot(x, rolling_means_max, '-k', linewidth='0.5', alpha=0.3)
plt.plot(x, rolling_means_mean, '-b', linewidth='0.5')
plt.fill_between(x[:], rolling_means_min[:], rolling_means_max[:], color='lightgray', alpha=0.3)
# plt.plot(x, rolling_means[3], '-b', linewidth='0.5', label='500 epi, vir 9')
# plt.plot(x, rolling_means[4], 'pink', linewidth='0.5', label='500 epi, vir 20')
# plt.plot(x, rolling_means[5], '-y', linewidth='0.5', label='virtual 9')
plt.legend()
plt.show()
