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

import pandas as pd
import numpy as np

from keras.models import load_model, Model
from input_data import getData

temp, data = getData()

input_tensor = data[0].reshape(1,4,4,4,1)

model = load_model('Results/L4/1D_model.h5')

layer_outputs = [layer.output for layer in model.layers]

activation_model = Model(inputs=model.input, outputs=layer_outputs)

activations = activation_model.predict(input_tensor)

print activations[1].shape
print activations[2].shape
print activations[3].shape
print activations[4].shape
print activations[5].shape
print activations[6].shape


fig, ax = plt.subplots(8,30,figsize=(10,10))

ax[0,0].matshow(data[0][:,:,:,0].reshape(16,4))
ax[1,0].matshow(data[0][:,:,:,0].reshape(16,4))
ax[2,0].matshow(data[0][:,:,:,0].reshape(16,4))
ax[3,0].matshow(data[0][:,:,:,0].reshape(16,4))
ax[4,0].matshow(data[0][:,:,:,0].reshape(16,4))
ax[5,0].matshow(data[0][:,:,:,0].reshape(16,4))
ax[6,0].matshow(data[0][:,:,:,0].reshape(16,4))
for i in range(29):
    ax[0,1+i].matshow(activations[1][0,:,:,:,i].reshape(16,4))
    ax[1,1+i].matshow(activations[2][0,:,:,:,i].reshape(16,4))
    ax[2,1+i].matshow(activations[3][0,:,:,:,i].reshape(4,2))

for i in range(15):
    ax[3, 8+i].matshow(activations[4][0,:,:,:,i].reshape(4,2))
    ax[4, 8+i].matshow(activations[5][0,:,:,:,i].reshape(4,2))
    ax[5, 8+i].matshow(activations[6][0,:,:,:,i].reshape(1,1))

ax[6,1].matshow(activations[7][0,:,:,:,:].reshape(15,3))
ax[7,1].matshow(activations[8][0,:,:,:,:].reshape(13,5))

for a in np.ravel(ax):
    a.grid(False)
    a.set_xticks([])
    a.set_yticks([])
plt.show()
