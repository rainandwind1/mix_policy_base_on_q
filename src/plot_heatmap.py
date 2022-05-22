import numpy as np
import random
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# plt.rcParams['font.sans-serif'] = ['FangSong']
# plt.rcParams['axes.unicode_minus']=False

def make_heat_map(path, begin, line_width = 0.003, patical = False):
    
    matrix = np.load(path)
    matrix = matrix.reshape((matrix.shape[1], matrix.shape[0], matrix.shape[2]))
    print(matrix.shape)
    if patical:
        plot_len = 10
    else:
        plot_len = matrix.shape[1]
    fig, ax = plt.subplots(1, matrix.shape[0], figsize = (7 * matrix.shape[0], plot_len), sharex="col")
    
    for i in range(matrix.shape[0]):
        if patical:
            sns.heatmap(matrix[i, begin:begin+plot_len, :], linewidths = line_width, ax = ax[i], square = True, annot = False)
        else:
            sns.heatmap(matrix[i], linewidths = line_width, ax = ax[i], square = True, annot = False)
        # ax[i].set_xticklabels(['移动', '敌人', '同伴', '自己'], rotation='horizontal')
        ax[i].set_xticklabels(['M', 'E', 'A', 'O'], rotation='horizontal')
        ax[i].set_yticklabels(range(begin, begin+plot_len), rotation='horizontal')
        ax[i].set_ylabel('Episode Step')
        ax[i].set_xlabel('Obs Feats')
    if patical:
        plt.savefig("./results/1025/{}_{}.svg".format(path[:-23], 'partical'))
    else:
        plt.savefig("./results/1025/{}.jpg".format(path[:-23]))


if __name__ == "__main__":
    make_heat_map("./3s_vs_4z_weights_17_10_2021.npy", 7, patical=True)
