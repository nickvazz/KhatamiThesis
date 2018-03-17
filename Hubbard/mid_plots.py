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

def middle_layer_plots(mid, train_output, test_output, y_train, y_test, U, layers):
    A,B,C,D = layers
    if mid == 2:
        x1, y1 = train_output[:,0], train_output[:,1]
        plt.scatter(x1,y1,c=y_train,s=1)
        x2, y2 = test_output[:,0], test_output[:,1]
        plt.scatter(x2,y2,c=y_test,s= 1)
        plt.colorbar()
        toFile = pd.DataFrame({'x':x1,'y':y1,'T':y_train})
        toFile.append(pd.DataFrame({'x':x2,'y':y2,'T':y_test}))
        toFile.sort_values('T',inplace=True)

        toFile.to_csv("Results/U{}/2D_ABCD{}_{}_{}_{}.csv".format(A,B,C,D), sep=',')
        plt.savefig("Results/U{}/2D_ABCD{}_{}_{}_{}.png".format(A,B,C,D))

        plt.clf()

        df = toFile
        plt.subplot(211)
        plt.hist2d(df.x,df.y,bins=25)
        plt.colorbar()
        plt.subplot(212)
        plt.scatter(df['x'],df['y'],c=df['T'],s=1)
        plt.colorbar()
        plt.savefig("Results/U{}/2D_Hist_ABCD{}_{}_{}_{}.png".format(U,A,B,C,D))
        # plt.show()
        plt.clf()
        model.reset_states()

    elif mid == 1:
        import matplotlib as mpl
        import matplotlib.cm as cm

        x1 = train_output[:,0]
        plt.scatter(y_train,x1,c=y_train,s=10)
        x2 = test_output[:,0]
        plt.scatter(y_test,x2,c=y_test,s=10)
        plt.colorbar()
        toFile = pd.DataFrame({'x':x1,'T':y_train})
        toFile.append(pd.DataFrame({'x':x2,'T':y_test}))
        toFile.to_csv("Results/U{}/1D_ABCD{}_{}_{}_{}.csv".format(U,A,B,C,D), sep=',')
        plt.savefig("Results/U{}/1D_ABCD{}_{}_{}_{}.png".format(U,A,B,C,D))
        plt.clf()

        norm = mpl.colors.Normalize(vmin=min(toFile['T']),vmax=max(toFile['T']))
        cmap = cm.viridis
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        for T in toFile['T'].unique():
            tempDF = toFile[toFile['T'] == T]
            plt.scatter(T, np.mean(tempDF['x']), c=m.to_rgba(T))
            plt.errorbar(T, np.mean(tempDF['x']), yerr=np.std(tempDF['x']), ecolor=m.to_rgba(T))

        plt.savefig("Results/U{}/1D_mean_ABCD{}_{}_{}_{}.png".format(U,A,B,C,D))
        # plt.show()
        plt.clf()


if __name__ == '__main__':
    from keras.models import load_model
    from input_data import getTempData
    from sklearn.model_selection import train_test_split
    from convolutional_autoencoder import model_creator
    from keras.models import Model

    mid = 2
    U = 5
    test = False
    num_pts = 100
    layers = [29,15,45,65]


    print 'why'
    data, temps = getTempData(num_data_points=num_pts,U=U, test=test)
    X_train, X_test, y_train, y_test = train_test_split(data, temps, test_size=.3, random_state=42, stratify=temps)
    print 'Results/U{}/{}D_model.json'.format(U,mid)
    # model = load_model('Results/U{}/{}D_model.json'.format(U,mid))
    model = model_creator(*layers)
    model.load_weights('Results/U{}/{}D_model.h5'.format(U,mid))
    print 'what'
    middle_layer = Model(inputs=model.input,
                         outputs=model.get_layer('code').output)
    print 'yea'
    train_output = middle_layer.predict(X_train)
    test_output = middle_layer.predict(X_test)

    middle_layer_plots(mid, train_output, test_output, y_train, y_test, U, layers)
