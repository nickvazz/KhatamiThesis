import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def getData(L=40, periodic=True,nDataPoints=1000):
    print 'hmm'
    if L == 40:
        # df = pd.DataFrame(pd.read_csv('isingData/2D40_training_data_50k.txt',header=None,sep='\s+',nrows=31*nDataPoints))
        # df = pd.DataFrame(pd.read_csv('isingData/2D40_training_data_50k.txt',header=None,sep='\s+'))

        # df = pd.DataFrame(pd.read_csv('isingData/2D40_training_data.txt',header=None,sep='\s+',nrows=101*nDataPoints))
        # df = pd.DataFrame(pd.read_csv('isingData/2D40_training_data.txt',header=None,sep='\s+'))
        df = pd.DataFrame(pd.read_csv('2D40_training_data_500.txt',header=None,sep='\s+'))
        print 'yup'
    else:
        print 'nope'
        pass

        # df = pd.DataFrame(pd.read_csv('isingData/2D' + str(L) + '_training_data.txt',header=None,sep='\s+'))
        # df = pd.DataFrame(pd.read_csv('isingData/2D' + str(L) + '_p_test_data.txt',header=None,sep='\s+'))
    # print len(df)
    # df = df[df[0]>=2]
    # df = df[df[0]<=3]

    temp = df[[0]].values.reshape(-1)
    print len(temp), 'datapoints'
    data = df[list(range(1,L**2+1))].values
    data = data.reshape(temp.shape[0],L,L)
    return temp, data
# temp, data = getData()


def dataInputPlot():
    f, ax = plt.subplots(8,10, figsize=(20,8))
    plt.suptitle('temp - index in dataset')
    for i in range(10):
        ax[0,i].imshow(data[i].reshape(40,40))
        ax[0,i].set_title(str(temp[i]) + '-' + str(i))
        ax[1,i].imshow(data[i+10].reshape(40,40))
        ax[1,i].set_title(str(temp[i+10]) + '-' + str(i+10))
        ax[2,i].imshow(data[i+20].reshape(40,40))
        ax[2,i].set_title(str(temp[i+20]) + '-' + str(i+20))
        ax[3,i].imshow(data[i+30].reshape(40,40))
        ax[3,i].set_title(str(temp[i+30]) + '-' + str(i+30))
        ax[4,i].imshow(data[i+40].reshape(40,40))
        ax[4,i].set_title(str(temp[i+40]) + '-' + str(i+40))
        ax[5,i].imshow(data[i+50].reshape(40,40))
        ax[5,i].set_title(str(temp[i+50]) + '-' + str(i+50))
        ax[6,i].imshow(data[i+60].reshape(40,40))
        ax[6,i].set_title(str(temp[i+60]) + '-' + str(i+60))
        ax[7,i].imshow(data[i+70].reshape(40,40))
        ax[7,i].set_title(str(temp[i+70]) + '-' + str(i+70))
    plt.show()
# dataInputPlot()
