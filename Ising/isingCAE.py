from ising_import_data import getData

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Flatten, Reshape
from keras.models import Model
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.initializers import he_normal
from keras import backend as K
from keras.callbacks import TensorBoard

from sklearn.model_selection import train_test_split

from datetime import datetime

import matplotlib as mpl
from sys import platform
if platform == 'darwin':
    mpl.use('TkAgg')
elif platform == "linux2":
    mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# TODO
# elu instead or relu

L = 40
temp, data = getData()
np.random.seed(42)
np.random.shuffle(data)
np.random.seed(42)
np.random.shuffle(temp)
# data=data[:1000]
# temp=temp[:1000]
data = data.reshape(len(data),L,L,1)

dropoutRate = 0.5
x_train, x_test, y_train, y_test = train_test_split(data, temp, test_size=.3, random_state=42)
# print x_train.shape

datagen = ImageDataGenerator(
            # width_shift_range=1,
            # height_shift_range=1,
            horizontal_flip=True,
            vertical_flip=True,
)

datagen.fit(x_train)

input_img = Input(shape=(L, L, 1))  # adapt this if using `channels_first` image data format
print K.int_shape(input_img), 'in'

x = Conv2D(16, (3, 3), kernel_initializer=he_normal(),activation='relu', strides=1, padding='same')(input_img)
# # x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
print K.int_shape(x), 'p1'

# # x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = Conv2D(8, (3, 3), kernel_initializer=he_normal(), activation='relu', strides=1, padding='same')(x)
# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
print K.int_shape(x), 'p2'

# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = Conv2D(8, (3, 3), kernel_initializer=he_normal(), activation='relu', strides=1, padding='same')(x)
# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = MaxPooling2D((2, 2), padding='same')(x)
print K.int_shape(x), 'p3'

x = Flatten()(x)
print K.int_shape(x), 'flatten'

# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = Dense(2, kernel_initializer=he_normal(), name='middle2in')(x)
# x = Dense(2, kernel_initializer=he_normal(), activation='relu', name='middle2in')(x)
print K.int_shape(x), 'mid2in'

# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = Dense(1, kernel_initializer=he_normal(),  name='middle1')(x)
# x = Dense(1, kernel_initializer=he_normal(), activation='relu', name='middle1')(x)
print K.int_shape(x), 'mid1'

# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = Dense(2, kernel_initializer=he_normal(), name='middle2out')(x)
# x = Dense(2, kernel_initializer=he_normal(), activation='relu', name='middle2out')(x)
print K.int_shape(x), 'mid2out'

# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = Dense(200, kernel_initializer=he_normal(), activation=None)(x)
print K.int_shape(x), 'unflatten'
x = Reshape([5,5,8])(x)
print K.int_shape(x), 'reshape'

# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = Conv2D(8, (3, 3), kernel_initializer=he_normal(), activation='relu', strides=1, padding='same')(x)
print K.int_shape(x), 'u1'
# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)

# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = Conv2D(8, (3, 3), kernel_initializer=he_normal(), activation='relu', strides=1, padding='same')(x)
print K.int_shape(x), 'u2'
# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)

# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), kernel_initializer=he_normal(), activation='relu', strides=1, padding='same')(x)
print K.int_shape(x), 'u3'
# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = UpSampling2D((2, 2))(x)
# print K.int_shape(x), 'u3'

# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
x = Conv2D(16, (3, 3), kernel_initializer=he_normal(), activation='relu', strides=1, padding='same')(x)
print K.int_shape(x), 'out'

# x = Dropout(dropoutRate)(x)
# x = BatchNormalization()(x)
decoded = Conv2D(1, (3, 3), kernel_initializer=he_normal(), activation='relu', strides=1, padding='same',name='output')(x)
print K.int_shape(decoded), 'out2'

autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    # loss='mse',
                    metrics=['accuracy','mae'])

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_logs"
logdir = "{}/run-{}/".format(root_logdir, now)

autoencoder.fit_generator(datagen.flow(x_train, x_train, batch_size=10),
                steps_per_epoch=len(x_train)/10,
                epochs=2,
                validation_data=(x_test, x_test),
                callbacks=[TensorBoard(log_dir=logdir)],
                verbose=1
                )


# autoencoder.fit(x_train, x_train, epochs=2, batch_size=10,
#                 shuffle=True, validation_data=(x_test, x_test),
#                 callbacks=[TensorBoard(log_dir=logdir)])

output_layer = Model(inputs=autoencoder.input,
                     outputs=autoencoder.get_layer('output').output)
output_layer_output = output_layer.predict(x_train)

f, ax = plt.subplots(4,10, figsize=(20,8))
for i in range(10):
    idx = np.random.randint(len(x_train))
    idx2 = np.random.randint(len(x_train))

    ax[0,i].imshow(x_train[idx].reshape(40,40))
    ax[0,i].set_title(str(y_train[idx]) + '-' + str(i))
    ax[1,i].imshow(output_layer_output[idx].reshape(40,40))
    ax[1,i].set_title(str(y_train[idx]) + '-' + str(i))

    ax[2,i].imshow(x_train[idx2].reshape(40,40))
    ax[2,i].set_title(str(y_train[idx2]) + '-' + str(i+10))
    ax[3,i].imshow(output_layer_output[idx2].reshape(40,40))
    ax[3,i].set_title(str(y_train[idx2]) + '-' + str(i+10))
# plt.colorbar()
plt.savefig('output{}.png'.format(now))
# plt.show()
plt.clf()

middle_layer = Model(inputs=autoencoder.input,
                     outputs=autoencoder.get_layer('middle2in').output)
middle_layer_output1 = middle_layer.predict(x_train)
middle_layer_output2 = middle_layer.predict(x_test)

x1, y1 = middle_layer_output1[:,0], middle_layer_output1[:,1]
plt.scatter(x1,y1,c=y_train,s=5)
x2, y2 = middle_layer_output2[:,0], middle_layer_output2[:,1]
plt.scatter(x2,y2,c=y_test,s=5)
toFile = pd.DataFrame({'x':x1,'y':y1,'T':y_train})
toFile.append(pd.DataFrame({'x':x2,'y':y2,'T':y_test}))
toFile.to_csv("isingMiddleData2D{}.csv".format(now), sep=',')
plt.savefig("dataGenIsing2d{}.png".format(now))
# plt.show()
plt.clf()

middle_layer = Model(inputs=autoencoder.input,
                     outputs=autoencoder.get_layer('middle1').output)
middle_layer_output1 = middle_layer.predict(x_train)[:,0]
middle_layer_output2 = middle_layer.predict(x_test)[:,0]
toFile = pd.DataFrame({'x':middle_layer_output1,'T':y_train})
toFile.append(pd.DataFrame({'x':middle_layer_output2,'T':y_test}))
plt.scatter(toFile['T'],toFile['x'],s=5,c=toFile['T'])
toFile.to_csv("isingMiddleData1D{}.csv".format(now), sep=',')
plt.savefig("dataGenIsing1d{}.png".format(now))

print(datetime.utcnow().strftime("%Y%m%d%H%M%S"))
