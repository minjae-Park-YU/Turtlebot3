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
dir_path = filedialog.askdirectory(parent=root,initialdir="C:/Users/user/anaconda3/envs/New2048RL/Codes",title='Please select a directory')
dir_path = dir_path + "/*"
file_list = glob.glob(dir_path)
file_names_csv = [file for file in file_list if file.endswith(".csv")]

color_list = ['b', 'g', 'r', 'k', 'y']
N = 10  # Rolling mean 간격 설정
datasets = []
data_min = []
data_max = []
data_mean = []
part_list = []

for i in range(len(file_names_csv)):
    datasets.append(pd.read_csv(file_names_csv[i]))

data_len = len(datasets[0]['reward'])
# 각각을 출력하기 위해 각 리스트에 저장
for i in range(len(datasets)):
    mini_list = []
    for j in range(data_len):
        mini_list.append(datasets[i]['reward'][j])
    part_list.append(mini_list)

for j in range(len(datasets[0]['reward'])):
    temp_list = []
    # 앙상블 하기위해 min, max, mean을 계산
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
# 각 그래프에 대한 Rolling mean 계산
mini_list1 = []
mini_list2 = []
color_index = 0

plt.subplot(121)
for i in range(len(file_list)):
    rm = [np.mean(part_list[i][p:p+N]) for p in range(data_len-N+1)]
    plt.plot(rm, color_list[color_index], linewidth='0.5')
    color_index += 1

x = np.arange(0, len(datasets[0]['reward'])-N+1)
plt.xlim([0, 1000])
plt.ylim([-400, 1000])

plt.subplot(122)
plt.plot(rolling_means_min, '-k', linewidth='0.5', alpha=0.3)
plt.plot(rolling_means_max, '-k', linewidth='0.5', alpha=0.3)
plt.plot(rolling_means_mean, '-b', linewidth='0.5')
plt.fill_between(x[:], rolling_means_min[:], rolling_means_max[:], color='lightgray', alpha=0.3)

plt.xlim([0, 1000])
plt.ylim([-400, 1000])
plt.show()
