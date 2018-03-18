import json
import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D, Flatten, Reshape
from keras.models import Model, Sequential
import keras.optimizers
from keras import backend as K
from input_data import getTempData
from mid_plots import middle_layer_plots
from metric_plots import make_metric_plot
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

def model_creator(A,B,C,D, mid):
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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae','accuracy'])
    return model

def train_model(U, num_pts, mid, epochs=50, test=False, make_plots=True, layers=[29,15,45,65]):
    A,B,C,D = layers

    np.random.seed(42)

    model = model_creator(A,B,C,D, mid)
    # print model.summary()

    # need graphviz (used homebrew)
    # need pydot and pydot-ng
    # from keras.utils import plot_model
    # plot_model(model, to_file='Results/{}D_model.png'.format(U,mid,mid),show_shapes=True)

    history = model.fit(X_train, X_train, epochs=epochs, batch_size=50,
                validation_data=(X_test, X_test),
                shuffle=True,
                verbose=2)
    metrics = dict(history.history.items())
    metrics['epochs'] = range(1,epochs+1)

    with open('Results/U{}/{}D_metrics.json'.format(U,mid,mid), 'w') as metric_file:
        metric_file.write(json.dumps(metrics))

    middle_layer = Model(inputs=model.input,
                         outputs=model.get_layer('code').output)
    train_output = middle_layer.predict(X_train)
    test_output = middle_layer.predict(X_test)


    if make_plots == True:
        middle_layer_plots(mid, train_output, test_output, y_train, y_test, U, layers)
        make_metric_plot(U,mid)

    model_json = model.to_json()
    with open("Results/U{}/{}D_model.json".format(U,mid,mid), "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights("Results/U{}/{}D_model.h5".format(U,mid,mid))
    print("Saved model to disk")

    model.reset_states()

if __name__ == '__main__':

    num_pts = 10
    epochs = 5
    test = True
    make_plots = True

    # for U in [5,6,8,9,10,12,14,16,20]:
    for U in [4]:

        data, temps = getTempData(num_data_points=num_pts,U=U, test=test, add_flip=True)
        X_train, X_test, y_train, y_test = train_test_split(data, temps, test_size=.3, random_state=42, stratify=temps)

        for mid in [1,2]:
            train_model(U, num_pts, mid, epochs=epochs, test=test, make_plots=make_plots)
