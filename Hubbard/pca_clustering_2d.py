import glob

import matplotlib as mpl
from sys import platform
if platform == 'darwin':
    mpl.use('TkAgg')
elif platform == "linux2":
    mpl.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns
sns.set()

from sklearn.decomposition import PCA

data = glob.glob('Results/U4/2D*.csv')[0]

df = pd.read_csv(data)[['T','x','y']]

X = df[['x','y']]

def pca_plots():
    cmap = mpl.colors.ListedColormap(sns.color_palette("RdBu_r", 31))

    fig, ax = plt.subplots(4, figsize=(8,12))
    df.plot(x='x',y='y',c='T',s=1, alpha=0.25,kind='scatter', cmap=cmap, ax=ax[0])
    ax[0].scatter(df['x'].mean(), df['y'].mean(), s=10,marker='*')

    pca2 = PCA(n_components=2)
    X_pca = pca2.fit_transform(X)

    df_pca = pd.DataFrame(X_pca, columns=None)
    df_pca = pd.concat([df_pca, df[['T']]], axis=1)
    df_pca.columns = ['x','y','T']
    df_pca['r'] = (df_pca['x']**2 + df_pca['y']**2)

    df_pca.plot(x='x',y='y',c='T',s=1,alpha=0.25,kind='scatter',cmap=cmap,ax=ax[1])

    pca1 = PCA(n_components=1)
    X_pca1 = pca1.fit_transform(X)
    df_pca1 = pd.DataFrame(X_pca1, columns=['x'])
    df_pca1 = pd.concat([df_pca1, df[['T']]], axis=1)

    # df_pca['r'] = (df_pca['x']**2 + df_pca['y']**2)

    for idx, T in enumerate(df_pca1['T'].unique()):
        temp_df = df_pca1[df_pca1['T'] == T].head()

        kde = sns.kdeplot(temp_df['x'], color=cmap.colors[idx],ax=ax[2], alpha=0.75)

        ax[2].legend_.remove()


    for idx, line in enumerate(kde.get_lines()):
        val_max = np.argmax(line.get_ydata())
        # print idx, np.argmax(line.get_ydata())
        y = line.get_ydata()[val_max]
        x = kde.get_lines()[idx].get_xdata()[val_max]
        ax[3].scatter(x,y,c=cmap.colors[idx])

    # print dir(kde.get_lines()[0])
    titles = ['2d','pca2d','kde of pca1d', 'max of kde of pca1d']
    for idx, axes in enumerate(ax):
        axes.set_title(titles[idx])
    plt.show()

def threeDpcaPlot():

    pca2 = PCA(n_components=2)
    X_pca = pca2.fit_transform(X)

    df_pca = pd.DataFrame(X_pca, columns=None)
    df_pca = pd.concat([df_pca, df[['T']]], axis=1)
    df_pca.columns = ['x','y','T']
    df_pca['r'] = (df_pca['x']**2 + df_pca['y']**2)

    fig, ax = plt.subplots(1)
    ax3D = fig.add_subplot(1,1,1, projection='3d')
    ax3D.scatter(df_pca['x'], df_pca['y'], df_pca['r'], s=1, c=df_pca['T'])
    ax3D.view_init(45,0)
    plt.show()

def box_violin_pca_plot():
    pca2 = PCA(n_components=2)
    X_pca = pca2.fit_transform(X)

    df_pca = pd.DataFrame(X_pca, columns=None)
    df_pca = pd.concat([df_pca, df[['T']]], axis=1)
    df_pca.columns = ['x','y','T']
    df_pca['r'] = (df_pca['x']**2 + df_pca['y']**2)

    plt.subplots(1, figsize=(10,5))
    sns.boxplot(x='T',y='r',data=df_pca)
    plt.show()

    plt.subplots(1, figsize=(10,5))
    sns.violinplot(x='T',y='r',data=df_pca)
    plt.show()

# pca_plots()
# threeDpcaPlot()
# box_violin_pca_plot()
