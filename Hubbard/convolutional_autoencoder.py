import json
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D, Flatten, Reshape
from keras.models import Model, Sequential
import keras.optimizers
from keras import backend as K
from input_data import getTempData
from sklearn.model_selection import train_test_split

import matplotlib as mpl
from sys import platform
if platform == 'darwin':
    mpl.use('TkAgg')
elif platform == "linux2":
    mpl.use('Agg')
import matplotlib.pyplot as plt

import seaborn as sns
from keras.callbacks import TensorBoard
import pandas as pd
from datetime import datetime

sns.set()

num_pts = 100
A,B,C,D = 29,15,45,65
mid = 2
epochs = 50
make_plots = True

print 'loading {} points per temp'.format(num_pts)

data, temps = getTempData(num_data_points=num_pts,U=4, test=True)
X_train, X_test, y_train, y_test = train_test_split(data, temps, test_size=.3, random_state=42, stratify=temps)

np.random.seed(42)

def model_creator(A,B,C,D):
    input_data = Input(shape=(4,4,4,200,))
    x = Conv3D(A,(2,2,2), padding='same', activation='relu')(input_data)
    # x = Conv3D(A,(2,2,2), padding='same', activation='relu')(x)
    x = Conv3D(A,(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = Conv3D(B,(2,2,2), padding='same', activation='relu')(x)
    # x = Conv3D(B,(2,2,2), padding='same', activation='relu')(x)
    x = Conv3D(B,(1,1,1), padding='same', activation='relu')(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    x = Conv3D(C,(1,1,1), padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(D)(x)
    x = Dense(mid, name='code')(x)
    x = Dense(D)(x)
    x = Reshape([1,1,1,D])(x)
    x = Conv3D(C,(1,1,1), padding='same', activation='relu')(x)
    x = UpSampling3D(size=(2,2,2))(x)
    x = Conv3D(B,(2,2,2), padding='same', activation='relu')(x)
    # x = Conv3D(B,(2,2,2), padding='same', activation='relu')(x)
    x = UpSampling3D(size=(2,2,2))(x)
    x = Conv3D(A,(2,2,2), padding='same', activation='relu')(x)
    # x = Conv3D(A,(2,2,2), padding='same', activation='relu')(x)
    # x = Conv3D(A,(1,1,1), padding='same', activation='relu')(x)
    output = Conv3D(200,(2,2,2), padding='same', activation='relu')(x)

    model = Model(input_data, output)
    return model

model = model_creator(A,B,C,D)
# print model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae','accuracy'])

# need graphviz (used homebrew)
# need pydot and pydot-ng
from keras.utils import plot_model
plot_model(model, to_file='Results/{}D/{}D_model.png'.format(mid,mid),show_shapes=True)

history = model.fit(X_train, X_train, epochs=epochs, batch_size=50,
            validation_data=(X_test, X_test),
            shuffle=True,
#             callbacks=[TensorBoard(log_dir=logdir)],
            verbose=1)
metrics = dict(history.history.items())
metrics['epochs'] = range(1,epochs+1)

with open('Results/{}D/{}D_metrics.json'.format(mid,mid), 'w') as metric_file:
    metric_file.write(json.dumps(metrics))

middle_layer = Model(inputs=model.input,
                     outputs=model.get_layer('code').output)
train_output = middle_layer.predict(X_train)
test_output = middle_layer.predict(X_test)

def middle_layer_plots(mid):
    if mid == 2:
        x1, y1 = train_output[:,0], train_output[:,1]
        plt.scatter(x1,y1,c=y_train,s=1)
        x2, y2 = test_output[:,0], test_output[:,1]
        plt.scatter(x2,y2,c=y_test,s= 1)
        plt.colorbar()
        toFile = pd.DataFrame({'x':x1,'y':y1,'T':y_train})
        toFile.append(pd.DataFrame({'x':x2,'y':y2,'T':y_test}))
        toFile.sort_values('T',inplace=True)

        toFile.to_csv("Results/2D/2D_ABCD{}_{}_{}_{}.csv".format(A,B,C,D), sep=',')
        plt.savefig("Results/2D/2D_ABCD{}_{}_{}_{}.png".format(A,B,C,D))

        plt.clf()

        df = toFile
        plt.subplot(211)
        plt.hist2d(df.x,df.y,bins=25)
        plt.colorbar()
        plt.subplot(212)
        plt.scatter(df['x'],df['y'],c=df['T'],s=1)
        plt.colorbar()
        plt.savefig("Results/2D/2D_Hist_ABCD{}_{}_{}_{}.png".format(A,B,C,D))
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
        toFile.to_csv("Results/1D/1D_ABCD{}_{}_{}_{}.csv".format(A,B,C,D), sep=',')
        plt.savefig("Results/1D/1D_ABCD{}_{}_{}_{}.png".format(A,B,C,D))
        plt.clf()

        norm = mpl.colors.Normalize(vmin=min(toFile['T']),vmax=max(toFile['T']))
        cmap = cm.viridis
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        for T in toFile['T'].unique():
            tempDF = toFile[toFile['T'] == T]
            plt.scatter(T, np.mean(tempDF['x']), c=m.to_rgba(T))
            plt.errorbar(T, np.mean(tempDF['x']), yerr=np.std(tempDF['x']), ecolor=m.to_rgba(T))

        plt.savefig("Results/1D/1D_mean_ABCD{}_{}_{}_{}.png".format(A,B,C,D))
        # plt.show()
        plt.clf()

if make_plots == True:
    middle_layer_plots(mid)

model.reset_states()
model_json = model.to_json()
with open("Results/{}D/{}D_model.json".format(mid,mid), "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("Results/{}D/{}D_model.h5".format(mid,mid))
print("Saved model to disk")
