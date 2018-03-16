import tensorflow as tf
import numpy as np
from keras.layers import Input, Dense, Conv3D, MaxPooling3D, UpSampling3D, Flatten, Reshape
from keras.models import Model, Sequential
import keras.optimizers
from keras import backend as K
from input_data import getTempData
from sklearn.model_selection import train_test_split
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard
import pandas as pd
from datetime import datetime

# import multiprocessing
# config = tf.ConfigProto().gpu_options.per_process_gpu_memory_fraction = 0.4
# import time
# def reshape_data(item):
#     return item.reshape(4,4,4,200).swapaxes(1,2).swapaxes(1,2)
# # cpus = multiprocessing.cpu_count()
# pool = multiprocessing.Pool(processes=cpus)
# # new_data = np.array(pool.map(reshape_data, data))



num_pts = 5
L = 4
print 'loading {} points per temp'.format(num_pts)

data, temps = getTempData(num_data_points=num_pts,U=4, test=True)

print ('loaded'), len(data)


data = data.reshape(len(data)/12800,12800)
data = np.asarray([item.reshape(4,4,4,200).swapaxes(1,2).swapaxes(1,2) for item in data])


X_train, X_test, y_train, y_test = train_test_split(data, temps, test_size=.3, random_state=42, stratify=temps)

print ('b')

np.random.seed(42)
A,B,C,D = 29,15,45,65
mid = 2
try:
    print 'c'
    if L == 4:
        input_data = Input(shape=(4,4,4,200,))
        x = Conv3D(A,(2,2,2), padding='same', activation='relu')(input_data)
        # x = Conv3D(A,(2,2,2), padding='same', activation='relu')(x)
        x = Conv3D(A,(1,1,1), padding='same', activation='relu')(x)
        # print K.int_shape(x)
        x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    if L == 8:
	print 'D'
        input_data = Input(shape=(8,8,8,200,))
        print K.int_shape(input_data)
        x = Conv3D(A,(2,2,2), padding='same', activation='relu')(input_data)
        print 'd1'
	# x = Conv3D(A,(2,2,2), padding='same', activation='relu')(x)
        x = Conv3D(A,(1,1,1), padding='same', activation='relu')(x)
        print 'd2'
        x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
        print K.int_shape(x)
        x = Conv3D(A,(2,2,2), padding='same', activation='relu')(x)
        # x = Conv3D(A,(2,2,2), padding='same', activation='relu')(x)
        x = Conv3D(A,(1,1,1), padding='same', activation='relu')(x)
        print K.int_shape(x)

        x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)

    print 'e'
    # print K.int_shape(x)
    x = Conv3D(B,(2,2,2), padding='same', activation='relu')(x)
    # x = Conv3D(B,(2,2,2), padding='same', activation='relu')(x)
    x = Conv3D(B,(1,1,1), padding='same', activation='relu')(x)
    # print K.int_shape(x)
    x = MaxPooling3D(pool_size=(2,2,2), padding='same')(x)
    # print K.int_shape(x)
    x = Conv3D(C,(1,1,1), padding='same', activation='relu')(x)
    # print K.int_shape(x)
    x = Flatten()(x)
    x = Dense(D)(x)
    # print K.int_shape(x)
    x = Dense(mid, name='code')(x)
    # print K.int_shape(x)
    x = Dense(D)(x)
    # print K.int_shape(x)
    x = Reshape([1,1,1,D])(x)
    # print K.int_shape(x)
    x = Conv3D(C,(1,1,1), padding='same', activation='relu')(x)
    # print K.int_shape(x)
    x = UpSampling3D(size=(2,2,2))(x)
    # print K.int_shape(x)
    x = Conv3D(B,(2,2,2), padding='same', activation='relu')(x)
    # x = Conv3D(B,(2,2,2), padding='same', activation='relu')(x)
    # print K.int_shape(x)
    x = UpSampling3D(size=(2,2,2))(x)
    # print K.int_shape(x)
    x = Conv3D(A,(2,2,2), padding='same', activation='relu')(x)
    # x = Conv3D(A,(2,2,2), padding='same', activation='relu')(x)
    # x = Conv3D(A,(1,1,1), padding='same', activation='relu')(x)
    if L == 4:
        output = Conv3D(200,(2,2,2), padding='same', activation='relu')(x)
    if L == 8:
	print 'f'
        x = Conv3D(B,(2,2,2), padding='same', activation='relu')(x)
        # x = Conv3D(B,(2,2,2), padding='same', activation='relu')(x)
        # print K.int_shape(x)
        x = UpSampling3D(size=(2,2,2))(x)
        output = Conv3D(200,(2,2,2), padding='same', activation='relu')(x)
    # pr
    # print K.int_shape(output)
    print K.int_shape(output)
    ConAE = Model(input_data, output)
    print ConAE.summary()

    ConAE.compile(optimizer='adam', loss='binary_crossentropy', metrics=['mae','accuracy'])
    from keras.utils import plot_model
    # plot_model(ConAE, to_file='model.png',show_shapes=True)
    # shit

    #now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    #root_logdir = "tf_logs"; logdir = "{}/run{}_ABCD{}_{}_{}_{}/".format(root_logdir, now,A,B,C,D)

    ConAE.fit(X_train, X_train, epochs=50, batch_size=50,
                validation_data=(X_test, X_test),
                shuffle=True,
   #             callbacks=[TensorBoard(log_dir=logdir)],
                verbose=1)



    middle_layer = Model(inputs=ConAE.input,
                         outputs=ConAE.get_layer('code').output)
    train_output = middle_layer.predict(X_train)
    test_output = middle_layer.predict(X_test)
    if mid == 2:

        x1, y1 = train_output[:,0], train_output[:,1]
        plt.scatter(x1,y1,c=y_train,s=1)
        x2, y2 = test_output[:,0], test_output[:,1]
        plt.scatter(x2,y2,c=y_test,s= 1)
        plt.colorbar()
        toFile = pd.DataFrame({'x':x1,'y':y1,'T':y_train})
        toFile.append(pd.DataFrame({'x':x2,'y':y2,'T':y_test}))
        toFile.to_csv("2D_ABCD{}_{}_{}_{}.csv".format(A,B,C,D), sep=',')
        plt.savefig("2D_ABCD{}_{}_{}_{}.png".format(A,B,C,D))
        # plt.show()
        plt.clf()

        df = toFile
        # ax.scatter(df['x'],df['y'],c=df['T'],s=0.1)
        plt.subplot(211)
        plt.hist2d(df.x,df.y,bins=25)
        plt.colorbar()
        plt.subplot(212)
        plt.scatter(df['x'],df['y'],c=df['T'],s=1)
        plt.colorbar()
        plt.savefig("Hist2D_ABCD{}_{}_{}_{}.png".format(A,B,C,D))
        # plt.show()
        plt.clf()
        ConAE.reset_states()

    elif mid == 1:

        x1 = train_output[:,0]
        plt.scatter(y_train,x1,c=y_train,s=10)
        x2 = test_output[:,0]
        plt.scatter(y_test,x2,c=y_test,s=10)
        plt.colorbar()
        toFile = pd.DataFrame({'x':x1,'T':y_train})
        toFile.append(pd.DataFrame({'x':x2,'T':y_test}))
        toFile.to_csv("1D_ABCD{}_{}_{}_{}.csv".format(A,B,C,D), sep=',')
        plt.savefig("1D_ABCD{}_{}_{}_{}.png".format(A,B,C,D))
        # plt.show()
        plt.clf()

        import matplotlib as mpl
        import matplotlib.cm as cm

        norm = mpl.colors.Normalize(vmin=min(toFile['T']),vmax=max(toFile['T']))
        cmap = cm.viridis
        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        for T in toFile['T'].unique():
            # print T
            tempDF = toFile[toFile['T'] == T]
            # print tempDF.head()
            plt.scatter(T, np.mean(tempDF['x']), c=m.to_rgba(T))
            plt.errorbar(T, np.mean(tempDF['x']), yerr=np.std(tempDF['x']), ecolor=m.to_rgba(T))

        plt.savefig("1D_mean_ABCD{}_{}_{}_{}.png".format(A,B,C,D))
        # plt.show()
        plt.clf()

        # df = toFile
        # ax.scatter(df['x'],df['y'],c=df['T'],s=0.1)
        # plt.subplot(211)
        # plt.hist2d(df.x,df.y,bins=25)
        # plt.colorbar()
        # plt.subplot(212)
        # plt.scatter(df['x'],df['y'],c=df['T'],s=1)
        # plt.colorbar()
        # plt.savefig("Hist2D_ABCD{}_{}_{}_{}.png".format(A,B,C,D))
        # # plt.show()
        # plt.clf()
        # ConAE.reset_states()
    # ConAE.save_weights('middle2D.h5')
    model_json = ConAE.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    ConAE.save_weights("model.h5")
    print("Saved model to disk")
except:
    print 'fail'
    pass
