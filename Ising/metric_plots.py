import matplotlib as mpl
from sys import platform
if platform == 'darwin':
    mpl.use('TkAgg')
elif platform == "linux2":
    mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

def make_metric_plot(L,mid):
    df = pd.DataFrame(pd.read_json('Results/L{}/{}D_metrics.json'.format(L,mid)))
    df.index = df.epochs
    df.drop('epochs', axis=1, inplace=True)


    fig, ax = plt.subplots(3, sharex=True)

    df[['acc','val_acc']].plot(ax=ax[0], title='Accuracy')
    df[['loss','val_loss']].plot(ax=ax[1], title='Loss')
    df[['mean_absolute_error','val_mean_absolute_error']].plot(ax=ax[2], title='Mean Absolute Error')
    for axes in ax: axes.legend(['Training','Validation'])

    plt.tight_layout()
    plt.savefig('Results/L{}/{}D_metrics.jpg'.format(L,mid))
    plt.clf()

if __name__ == '__main__':

    for L in [4,8]:
        for mid in [1,2]:
            print L, mid
            make_metric_plot(L, mid)
