import matplotlib as mpl
from sys import platform
if platform == 'darwin':
    mpl.use('TkAgg')
elif platform == "linux2":
    mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

sns.set()

from keras.models import load_model, Model
from keras import backend as K
from input_data import getData

def max_activation(layer_output, filter_index, epochs, L=4, mid=1,physics='Ising'):

    np.random.seed(42)

    shape = layer_output.shape.as_list()

    if len(layer_output.shape) == 5:
        filter_output = layer_output[:,:,:,:,filter_index]
    elif len(layer_output.shape) == 4:
        filter_output = layer_output[:,:,:,filter_index]
    elif len(layer_output.shape) == 3:
        filter_output = layer_output[:,:,filter_index]
    elif len(layer_output.shape) == 2:
        filter_output = layer_output[:,filter_index]
    elif len(layer_output.shape) == 1:
        filter_output = layer_output[filter_index]

    loss = K.mean(filter_output)
    grads = K.gradients(loss, model.input)[0]
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    iterate = K.function([model.input], [loss, grads])

    if physics == 'Ising':
        loss_value, grads_value = iterate([np.zeros((1,L,L,L,1))])
        input_img_data = np.random.random((1,L,L,L,1)) * 20 + 128.
    if physics == 'Hubbard':
        loss_value, grads_value = iterate([np.zeros((1,L,L,L,200))])
        input_img_data = np.random.random((1,L,L,L,200)) * 20 + 128.

    step = 1.
    for i in range(epochs):
        loss_value, grads_value = iterate([input_img_data])
        input_img_data += grads_value * step

    return input_img_data






if __name__ == "__main__":
    physics = 'Ising'
    L = 4
    mid = 1

    for L in [4,8]:
        for mid in [1,2]:
            model = load_model('Results/L{}/{}D_model.h5'.format(L,mid))

            layer_outputs = [layer.output for layer in model.layers[1:]]

            for layer_output in layer_outputs:
                try:
                    shape = layer_output.shape.as_list()

                    plots = shape[-1]**0.5 % 1
                    if plots >= 0.5:
                        num_rows = num_cols = int(np.ceil(shape[-1]**0.5))
                    elif plots > 0:
                        num_cols = int(np.ceil(shape[-1]**0.5))
                        num_rows = num_cols - 1
                    else:
                        num_cols = num_rows = 1

                    fig, ax = plt.subplots(num_rows, num_cols, sharex=True, sharey=True)

                    layer_title = layer_output.name.split('/')[0]

                    plt.suptitle('{} Model Middle Layer Size = {} \nMax Activation Inputs for {} Filters'.format(physics,mid,layer_title))
                    for i in range(shape[-1]):
                        max_activation_output = max_activation(layer_output,i,3,L=L,mid=mid,physics=physics)

                        row = i // num_cols
                        col = i % num_cols

                        ax[row,col].matshow(np.ravel(max_activation_output).reshape(4,16))
                        ax[row,col].text(0,-2,'{}'.format(layer_title))
                        ax[row,col].text(0,8,'filter:{}'.format(i))

                    for a in np.ravel(ax):
                        a.set_xticks([])
                        a.set_yticks([])

                    plt.savefig('Results/max_activations/{}D_L{}_{}'.format(mid,L,layer_title))
                    # plt.show()
                except:
                    pass
