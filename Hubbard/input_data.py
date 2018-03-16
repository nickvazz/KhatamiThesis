import numpy as np
import glob
import pandas as pd
import multiprocessing


def getTempData(U=4, num_data_points=5, test=False):
    files = glob.glob('Data/4x4x4_Mu0/*U{}*/*'.format(U))
    files = sorted(files)
    if test == True:
        files = files[:3]

    # num_data_points = 5 # only for testing
    data = np.asarray([])
    temps = np.asarray([])
    for f in files:
        df = pd.DataFrame(pd.read_csv(f,header=None))[:num_data_points]

        data = np.append(data,np.array([np.array(list(item)[:12800], dtype=int ) for item in df[0]]))

        label_temp = f.split('_')[-1].split('.')[1]
        temperture = float('0.' + label_temp)

        temps = np.append(temps, np.ones(num_data_points) * temperture)
        data = data.reshape(len(data)/12800,12800)
        data = np.asarray([item.reshape(4,4,4,200).swapaxes(1,2).swapaxes(1,2) for item in data])


    return data, temps

if __name__ == '__main__':
    data, temps = getTempData(U=4, num_data_points=2, test=True)
    print temps.shape
    print data.shape
    print set(temps)
    print len(data)
