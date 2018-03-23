import matplotlib as mpl
from sys import platform
if platform == 'darwin':
    mpl.use('TkAgg')
elif platform == "linux2":
    mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

import pandas as pd
import numpy as np


def middle_layer_plots(mid, train_output, test_output, y_train, y_test, L, layers):
    A,B,C,D = layers
    if mid == 2:
        fig, ax = plt.subplots(1)
        x1, y1 = train_output[:,0], train_output[:,1]
        ax.scatter(x1,y1,c=y_train,s=1)
        x2, y2 = test_output[:,0], test_output[:,1]
        cax = ax.scatter(x2,y2,c=y_test,s= 1)
        fig.colorbar(cax)
        toFile = pd.DataFrame({'x':x1,'y':y1,'T':y_train})
        toFile.append(pd.DataFrame({'x':x2,'y':y2,'T':y_test}))
        toFile.sort_values('T',inplace=True)

        toFile.to_csv("Results/L{}/2D_ABCD{}_{}_{}_{}.csv".format(L,A,B,C,D), sep=',')
        plt.savefig("Results/L{}/2D_ABCD{}_{}_{}_{}.png".format(L,A,B,C,D))

        plt.clf()

        df = toFile
        fig, ax = plt.subplots(2,sharex=True)
        cax = ax[0].hist2d(df.x,df.y,bins=25)
        fig.colorbar(cax[3], ax=ax[0])
        cax1 = ax[1].scatter(df['x'],df['y'],c=df['T'],s=1)
        fig.colorbar(cax1)
        plt.savefig("Results/L{}/2D_Hist_ABCD{}_{}_{}_{}.png".format(L,A,B,C,D))
        # plt.show()
        plt.clf()


    elif mid == 1:
        import matplotlib as mpl
        import matplotlib.cm as cm
        from matplotlib.colors import ListedColormap
        fig, ax = plt.subplots(1)
        x1 = train_output[:,0]
        ax.scatter(y_train,x1,c=y_train,s=10)
        x2 = test_output[:,0]
        cax = ax.scatter(y_test,x2,c=y_test,s=10)
        fig.colorbar(cax)
        toFile = pd.DataFrame({'x':x1,'T':y_train})
        toFile.append(pd.DataFrame({'x':x2,'T':y_test}))
        toFile.to_csv("Results/L{}/1D_ABCD{}_{}_{}_{}.csv".format(L,A,B,C,D), sep=',')
        plt.savefig("Results/L{}/1D_ABCD{}_{}_{}_{}.png".format(L,A,B,C,D))
        plt.clf()

        norm = mpl.colors.Normalize(vmin=min(toFile['T']),vmax=max(toFile['T']))
        cmap = ListedColormap(sns.color_palette().as_hex())
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        sns.violinplot(x=toFile['T'],y=toFile['x'],inner='quartiles')
        plt.savefig("Results/L{}/1D_violin_ABCD{}_{}_{}_{}.png".format(L,A,B,C,D))
        plt.clf()
        sns.boxplot(x=toFile['T'],y=toFile['x'])
        plt.savefig("Results/L{}/1D_box_ABCD{}_{}_{}_{}.png".format(L,A,B,C,D))


        # plt.show()
        plt.clf()
