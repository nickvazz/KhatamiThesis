import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set()

mid = 1


df = pd.DataFrame(pd.read_json('Results/{}D/{}D_metrics.json'.format(mid,mid)))
df.index = df.epochs
df.drop('epochs', axis=1, inplace=True)




ax = df[['acc','val_acc']].plot()
plt.show()
ax = df[['loss','val_loss']].plot()
plt.show()
ax = df[['mean_absolute_error','val_mean_absolute_error']].plot()
plt.show()
