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

def getData(num_data_points=5, test=True,L=4):

    df = pd.DataFrame(pd.read_csv('Data/3D{}_training_data.txt'.format(L),header=None,sep='\s+', nrows=num_data_points*301))
    temp = df.loc[:,0].values
    data = df.iloc[:,1:L**3+1].values.reshape(temp.shape[0],L,L,L,1)
    return temp, data



if __name__ == '__main__':
    temp, data = getData(test=True, L=8)
    print temp.shape
    print data.shape
    print
